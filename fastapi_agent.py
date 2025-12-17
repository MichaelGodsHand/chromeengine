"""
FastAPI entry point for browser-use agent.
Accepts URL and action, maintains browser session state between requests.
Includes Daily.co streaming capabilities.
"""

import asyncio
import logging
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatOpenAI
from browser_use.agent.views import AgentHistoryList

# Add daily_streaming to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "daily_streaming"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Browser-Use Agent API",
    description="API for browser automation using browser-use with OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active agent sessions
# In production, use Redis or a database
active_sessions: dict[str, dict] = {}

# Agent configuration - easily toggle judge evaluation
USE_JUDGE = False  # Set to True to enable judge evaluation of agent tasks

# Headless mode configuration (for Cloud Run / production)
# Automatically set to True in Cloud Run, or set HEADLESS_MODE=true in .env
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "false").lower() == "true"
logger.info(f"🖥️ Headless mode: {HEADLESS_MODE}")

# Cloud Run API URL (hardcoded for bot to connect)
CLOUD_RUN_API_URL = "https://chromeengine-739298578243.us-central1.run.app"
logger.info(f"🔗 Cloud Run API URL: {CLOUD_RUN_API_URL}")

# Chrome args for Docker/Cloud Run environments
CHROME_ARGS = [
    '--no-sandbox',  # Required for Docker
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',  # Overcome limited resource problems
    '--disable-gpu',  # Disable GPU acceleration
    '--disable-software-rasterizer',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-sync',
    '--metrics-recording-only',
    '--no-first-run',
    '--safebrowsing-disable-auto-update',
    '--disable-blink-features=AutomationControlled',
] if HEADLESS_MODE else []  # Only apply these flags in headless/Cloud Run mode


# ============================================================================
# Models for Daily Streaming
# ============================================================================

class CreateRoomRequest(BaseModel):
    """Request model for creating a Daily room"""
    session_id: str
    room_name: Optional[str] = None


class CreateRoomResponse(BaseModel):
    """Response model for room creation"""
    room_url: str
    room_name: str
    session_id: str


class ScreenshotResponse(BaseModel):
    """Response model for screenshot"""
    screenshot: str
    session_id: str


# ============================================================================
# Models for Agent Actions
# ============================================================================

class ActionRequest(BaseModel):
    """Request model for agent actions"""
    url: str
    action: str
    session_id: Optional[str] = None  # Optional: if provided, continues existing session
    max_steps: int = 20  # Maximum steps for this action


class ActionResponse(BaseModel):
    """Response model for agent actions"""
    session_id: str
    success: bool
    result: str
    steps_taken: int
    urls_visited: list[str]
    error: Optional[str] = None


class SessionInfo(BaseModel):
    """Information about an active session"""
    session_id: str
    current_url: str
    total_steps: int
    created_at: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Browser-Use Agent API",
        "endpoints": {
            "POST /action": "Execute an action on a URL",
            "GET /sessions": "List active sessions",
            "DELETE /sessions/{session_id}": "Close a session"
        }
    }


@app.post("/action", response_model=ActionResponse)
async def execute_action(request: ActionRequest):
    """
    Execute an action on a URL.
    
    If session_id is provided, continues with existing browser session.
    Otherwise, creates a new session and navigates to the URL first.
    """
    try:
        # Validate URL
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
        
        # Check if OpenAI API key is set
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is not set"
            )
        
        # Get or create session
        session_id = None
        browser = None
        is_new_session = False
        
        # Check if session exists
        if request.session_id and request.session_id in active_sessions:
            # Continue existing session - reuse the SAME browser
            session_data = active_sessions[request.session_id]
            browser = session_data["browser"]
            session_id = request.session_id
            logger.info(f"✅ Continuing existing session {session_id[:8]}...")
            
            # Check if browser is still alive and connected
            try:
                current_url = await browser.get_current_page_url()
                logger.info(f"✅ Browser is alive and connected. Current URL: {current_url}")
                # Browser is good, no need to navigate - work on current page
            except Exception as e:
                logger.warning(f"⚠️ Browser session lost ({e}), will recreate...")
                # Browser died, create new one
                browser = Browser(
                    headless=HEADLESS_MODE,
                    window_size={'width': 1280, 'height': 720},
                    keep_alive=True,
                    args=CHROME_ARGS,
                )
                await browser.start()
                session_data["browser"] = browser
                is_new_session = True  # Need to navigate since browser was recreated
        else:
            # Create new session
            if request.session_id:
                # User provided session_id but it doesn't exist - create new one with that ID
                session_id = request.session_id
                logger.info(f"⚠️ Session ID {session_id[:8]} not found, creating new session with this ID...")
            else:
                # Generate new session ID
                session_id = str(uuid.uuid4())
                logger.info(f"🆕 Creating new session {session_id[:8]}...")
            
            is_new_session = True
            
            # Initialize browser
            browser = Browser(
                headless=HEADLESS_MODE,  # Auto-detects from environment
                window_size={'width': 1280, 'height': 720},
                keep_alive=True,  # Keep browser alive between requests
                args=CHROME_ARGS,  # Add Docker/Cloud Run specific flags
            )
            
            # Start browser session ONCE - this creates the browser window
            logger.info(f"🚀 Starting browser for session {session_id[:8]}...")
            logger.info(f"🔍 Browser instance ID: {id(browser)}")
            await browser.start()
            logger.info(f"✅ Browser started for session {session_id[:8]}")
            
            # Store session immediately to prevent duplicate creation
            active_sessions[session_id] = {
                "browser": browser,
                "created_at": asyncio.get_event_loop().time(),
                "total_steps": 0,
            }
            logger.info(f"💾 Session {session_id[:8]} stored in memory")
        
        # Only navigate if this is a NEW session
        if is_new_session:
            logger.info(f"🧭 Navigating to {request.url} for new session {session_id[:8]}...")
            # Initialize LLM (OpenAI only)
            llm = ChatOpenAI(model='gpt-4o')
            
            # Navigate to the URL first - use the EXISTING browser instance
            # Use flash_mode and simple task - just navigate, no summaries
            initial_task = f"Navigate to {request.url}. After navigating, IMMEDIATELY call done. Do NOT verify, do NOT check if it worked. Just navigate and call done."
            
            # Override system message completely to force immediate done after navigation
            navigation_system_message = """You are a navigation-only agent. Your job is simple:
1. Navigate to the requested URL
2. IMMEDIATELY call done after navigating
3. Do NOT verify if navigation worked
4. Do NOT extract content or provide summaries
5. Do NOT evaluate page contents

After navigating, you MUST call done in the same step or the very next step. Navigation is complete once you've navigated - no verification needed."""
            
            nav_agent = Agent(
                task=initial_task,
                llm=llm,
                browser=browser,  # Use the SAME browser instance that's already started
                flash_mode=True,  # Skip evaluation and thinking - just navigate
                override_system_message=navigation_system_message,  # Completely override default behavior
                max_actions_per_step=1,  # Only allow one action per step
                use_judge=USE_JUDGE,  # Enable/disable judge evaluation
            )
            
            # Run initial navigation - browser is already started, so this won't create a new one
            # Use low max_steps to force quick completion
            nav_history = await nav_agent.run(max_steps=3)
            active_sessions[session_id]["total_steps"] = nav_history.number_of_steps() if nav_history else 0
            logger.info(f"✅ Session {session_id[:8]} navigated to {request.url}")
        else:
            logger.info(f"⏭️ Skipping navigation - continuing on current page for session {session_id[:8]}")
        
        # Now execute the action using the SAME browser instance
        # Get current URL to provide context
        try:
            current_url = await browser.get_current_page_url()
            logger.info(f"📍 Current page URL: {current_url}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get current URL: {e}")
            current_url = request.url
        
        # Check if action is redundant (e.g., "go to this page" when already there after navigation)
        action_lower = request.action.lower().strip()
        is_redundant_navigation = (
            is_new_session and  # We just navigated
            any(phrase in action_lower for phrase in [
                "go to this page", "navigate to", "go to", "open this page", 
                "simply go", "just go", "visit this page"
            ]) and
            current_url == request.url  # Already on the requested URL
        )
        
        if is_redundant_navigation:
            # Skip creating a new agent - we're already where we need to be
            logger.info(f"⏭️ Skipping redundant action - already on {current_url}")
            success = True
            steps_taken = 0
            urls_visited = [current_url]
        else:
            # Create action task - navigation only, no summaries
            # Check if user explicitly said "do NOTHING else" or similar - if so, stop immediately after action
            action_lower = request.action.lower()
            must_stop_immediately = any(phrase in action_lower for phrase in [
                "do nothing else", "do not do anything else", "stop after", "only", "just", 
                "nothing else", "then stop", "and stop"
            ])
            
            if is_new_session:
                if must_stop_immediately:
                    action_task = f"You are already on {current_url}. {request.action}. CRITICAL: After performing the requested action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Just perform the action and call done."
                else:
                    action_task = f"You are already on {current_url}. {request.action}. IMPORTANT: If this involves a dropdown, follow these steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. After clicking a link/button that should navigate, wait for the URL to change and page to load before calling done. Do NOT call done until navigation completes. Do NOT verify content."
            else:
                if must_stop_immediately:
                    action_task = f"{request.action}. CRITICAL: After performing the requested action, IMMEDIATELY call done. Do NOT perform any other actions. Do NOT click anything else. Do NOT explore. Just perform the action and call done."
                else:
                    action_task = f"{request.action}. IMPORTANT: If this involves a dropdown, follow these steps: 1) Click to open the dropdown, 2) Use the 'wait' action to wait 3 seconds, 3) Then click the item inside the dropdown. After clicking a link/button that should navigate, wait for the URL to change and page to load before calling done. Do NOT call done until navigation completes. Do NOT verify content."
            
            # Create a new agent instance for this action (reuses the SAME browser)
            # CRITICAL: Pass the same browser instance - Agent will reuse it, not create a new one
            llm = ChatOpenAI(model='gpt-4o')
            
            # Verify we're using the stored browser instance
            stored_browser = active_sessions[session_id]["browser"]
            if browser is not stored_browser:
                logger.error(f"❌ Browser instance mismatch! Using stored browser instead.")
                browser = stored_browser
            
            logger.info(f"🤖 Creating Agent for action (browser instance ID: {id(browser)})...")
            
            # Override system message - handle dropdowns and explicit stop instructions
            if must_stop_immediately:
                navigation_system_message = """You are a navigation-only agent with STRICT stop instructions. Your job is:
1. Perform ONLY the requested action (click, navigate, scroll, etc.)
2. IMMEDIATELY call done after performing the action
3. Do NOT perform any other actions
4. Do NOT click anything else
5. Do NOT explore the page
6. Do NOT verify content
7. Do NOT extract information

CRITICAL RULES:
- If the task says "do NOTHING else" or "only" or "just", you MUST stop immediately after the action
- After clicking a button/link, if navigation occurs, wait for URL to change, then IMMEDIATELY call done
- After clicking a button/link, if NO navigation occurs, IMMEDIATELY call done
- Do NOT perform any additional clicks, scrolling, or exploration"""
            else:
                navigation_system_message = """You are a navigation-only agent. Your job is simple:
1. Perform the requested navigation action (click, navigate, scroll, etc.)
2. For dropdown menus: Follow these EXACT steps:
   a) Click to open the dropdown
   b) Use the 'wait' action to wait 3 seconds (this ensures the dropdown is fully visible)
   c) Then click the item inside the dropdown
3. If the action triggers navigation (clicking a link/button that navigates), wait for the page to load (URL changes) before calling done
4. If the action does NOT trigger navigation (like scrolling), call done immediately after the action
5. Do NOT verify content or extract information
6. Do NOT repeat the action unnecessarily
7. Do NOT provide summaries

IMPORTANT: 
- ALWAYS use 'wait' action for 3 seconds between opening a dropdown and clicking an item inside it
- When clicking links or navigation buttons, wait for the page to load (you'll see the URL change) before calling done
- If navigation doesn't happen after clicking, try clicking again or check if the element is actually clickable"""
            
            action_agent = Agent(
                task=action_task,
                llm=llm,
                browser=browser,  # Reuse the SAME browser instance - this is critical!
                flash_mode=True,  # Skip evaluation and thinking - just navigate
                override_system_message=navigation_system_message,  # Completely override default behavior
                max_actions_per_step=1 if must_stop_immediately else 3,  # Limit to 1 action if user said "do NOTHING else"
                use_judge=True,  # Enable judge for step-by-step verification
                judge_llm=llm,  # Use same LLM for judge
            )
            logger.info(f"✅ Agent created with browser session ID: {action_agent.browser_session.id}")
            
            # Shared flag to track if goal was achieved (used between callbacks)
            goal_achieved_flag = {"achieved": False, "step": None}
            
            # Check goal at START of each step (before actions execute)
            async def check_goal_before_step(agent):
                """At the start of each step, check if goal was achieved in previous step"""
                # If goal was achieved in previous step, force done immediately
                if goal_achieved_flag["achieved"]:
                    logger.info(f"🛑 Goal was achieved in step {goal_achieved_flag['step']}, forcing done at start of step {agent.state.n_steps}")
                    
                    # Use add_new_task to inject urgent message to call done
                    completion_message = (
                        f"URGENT: The judge verified in step {goal_achieved_flag['step']} that the task goal '{request.action}' has been achieved. "
                        f"You MUST call done immediately with success=True. Do NOT perform any actions. Just call done."
                    )
                    agent.add_new_task(completion_message)
            
            # Check goal at END of each step (after actions execute)
            async def check_goal_after_step(agent):
                """After each step, use judge to verify if the goal is achieved by checking the screen"""
                logger.info(f"🔍 Judge callback triggered after step {agent.state.n_steps}")
                
                # Check even if agent called done - we want to verify it was correct
                agent_called_done = agent.history.is_done()
                
                try:
                    # Get current browser state and screenshot
                    browser_state = await agent.browser_session.get_browser_state_summary()
                    current_url = browser_state.url
                    
                    # Get recent screenshots (last one is most recent)
                    screenshot_paths = [p for p in agent.history.screenshot_paths() if p is not None]
                    if not screenshot_paths:
                        logger.debug(f"⏭️ Skipping judge check at step {agent.state.n_steps} - no screenshots yet")
                        return  # No screenshots yet, skip check
                    
                    # Get current state summary
                    current_result = agent.history.final_result() or f"Current URL: {current_url}"
                    current_steps = agent.history.agent_steps()
                    
                    # Use judge to check if goal is achieved by examining the screen
                    from browser_use.agent.judge import construct_judge_messages
                    from browser_use.agent.views import JudgementResult
                    
                    # Use the most recent screenshot to check current state
                    recent_screenshots = screenshot_paths[-3:] if len(screenshot_paths) > 3 else screenshot_paths
                    
                    judge_messages = construct_judge_messages(
                        task=action_task,  # Original task to verify against
                        final_result=current_result,
                        agent_steps=current_steps[-5:] if len(current_steps) > 5 else current_steps,  # Last 5 steps
                        screenshot_paths=recent_screenshots,
                        max_images=3,  # Use last 3 screenshots
                        ground_truth=None,
                    )
                    
                    # Call judge LLM to verify goal achievement
                    kwargs: dict = {'output_format': JudgementResult}
                    if agent.judge_llm.provider == 'browser-use':
                        kwargs['request_type'] = 'judge'  # type: ignore
                    
                    logger.info(f"🔍 Judge checking if goal achieved after step {agent.state.n_steps} (agent called done: {agent_called_done})...")
                    response = await agent.judge_llm.ainvoke(judge_messages, **kwargs)
                    judgement: JudgementResult = response.completion  # type: ignore[assignment]
                    
                    # If judge says task is complete, set flag for next step
                    if judgement and judgement.verdict is True:
                        logger.info(f"✅ Judge verified goal achieved at step {agent.state.n_steps}! Will force done in next step.")
                        goal_achieved_flag["achieved"] = True
                        goal_achieved_flag["step"] = agent.state.n_steps
                        
                        # If user said "do NOTHING else", also inject message immediately to stop
                        if must_stop_immediately:
                            logger.info(f"🛑 User requested 'do NOTHING else' - injecting stop message immediately")
                            stop_message = (
                                f"URGENT: The judge verified that the task '{request.action}' is complete. "
                                f"You MUST call done immediately. Do NOT perform any more actions. "
                                f"The user explicitly said 'do NOTHING else' - stop now."
                            )
                            agent.add_new_task(stop_message)
                    else:
                        logger.info(f"⏳ Judge says goal not yet achieved at step {agent.state.n_steps} (verdict: {judgement.verdict if judgement else 'None'}), continuing...")
                        # If agent called done but judge says not achieved, log warning
                        if agent_called_done and judgement and judgement.verdict is False:
                            logger.warning(f"⚠️ Agent called done at step {agent.state.n_steps}, but judge says goal not achieved!")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Judge check after step {agent.state.n_steps} failed: {e}", exc_info=True)
            
            # Execute action - browser is already started, Agent.run() will reuse it
            # Use the user's requested max_steps without capping it
            logger.info(f"▶️ Executing action on session {session_id[:8]}: {request.action} (max {request.max_steps} steps)")
            history: AgentHistoryList = await action_agent.run(
                max_steps=request.max_steps,
                on_step_start=check_goal_before_step,  # Check if goal achieved before step starts
                on_step_end=check_goal_after_step  # Check goal after each step
            )
            logger.info(f"✅ Action completed for session {session_id[:8]}")
            
            # Get results from history
            success = history.is_done() and history.is_successful() is not False
            steps_taken = history.number_of_steps()
            urls_visited = [url for url in history.urls() if url is not None]
        
        # Get current URL from browser after action
        try:
            final_url = await browser.get_current_page_url()
        except:
            final_url = urls_visited[-1] if urls_visited else current_url
        
        # For navigation-only, just confirm the action was completed
        if is_redundant_navigation:
            result = f"Already on {final_url} - action skipped"
        else:
            result = f"Navigation completed to {final_url}" if success else "Navigation failed"
        
        # Update session data (session_id is already set above)
        if session_id in active_sessions:
            active_sessions[session_id]["total_steps"] = active_sessions[session_id].get("total_steps", 0) + steps_taken
        
        return ActionResponse(
            session_id=session_id,
            success=success,
            result=result,
            steps_taken=steps_taken,
            urls_visited=urls_visited,
        )
        
    except Exception as e:
        logger.error(f"Error executing action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, data in active_sessions.items():
        try:
            current_url = await data["browser"].get_current_page_url()
        except:
            current_url = "Unknown"
        
        sessions.append(SessionInfo(
            session_id=session_id,
            current_url=current_url,
            total_steps=data.get("total_steps", 0),
            created_at=str(data.get("created_at", "Unknown"))
        ))
    
    return sessions


# ============================================================================
# Daily.co Streaming Endpoints
# ============================================================================

@app.post("/streaming/create-room", response_model=CreateRoomResponse)
async def create_daily_room(request: CreateRoomRequest):
    """
    Create a Daily.co room for streaming a browser session.
    Also starts a Pipecat bot to stream the browser video.
    
    Requires DAILY_API_KEY environment variable to be set.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from daily_service import get_daily_service
        from browser_bot import start_bot_for_session
        
        daily_service = get_daily_service()
        if not daily_service:
            raise HTTPException(
                status_code=500,
                detail="Daily.co service not configured. Set DAILY_API_KEY environment variable."
            )
        
        # Create room
        room_data = await daily_service.create_room(
            name=request.room_name,
            privacy="public"
        )
        
        # Store room info in session
        active_sessions[request.session_id]["daily_room"] = {
            "url": room_data["url"],
            "name": room_data["name"]
        }
        
        logger.info(f"✅ Created Daily room for session {request.session_id[:8]}: {room_data['name']}")
        
        # Start Pipecat bot to stream browser video
        logger.info(f"🤖 Starting browser streaming bot...")
        bot_id = await start_bot_for_session(
            session_id=request.session_id,
            room_url=room_data["url"],
            fastapi_url=CLOUD_RUN_API_URL
        )
        
        active_sessions[request.session_id]["bot_id"] = bot_id
        logger.info(f"✅ Bot started: {bot_id}")
        
        return CreateRoomResponse(
            room_url=room_data["url"],
            room_name=room_data["name"],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error creating Daily room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streaming/get-screenshot/{session_id}", response_model=ScreenshotResponse)
async def get_screenshot(session_id: str):
    """
    Get the latest screenshot from a browser session.
    
    Returns base64-encoded PNG image data.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        browser = session_data["browser"]
        
        # Get current page
        page = await browser.get_current_page()
        if not page:
            raise HTTPException(status_code=500, detail="No active page in browser")
        
        # Capture screenshot (returns base64 string)
        screenshot_b64 = await page.screenshot()
        
        return ScreenshotResponse(
            screenshot=screenshot_b64,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/streaming/end-stream/{session_id}")
async def end_stream(session_id: str):
    """
    End a streaming session and cleanup Daily room and bot.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        
        # Stop bot if it exists
        if "bot_id" in session_data:
            try:
                from browser_bot import stop_bot
                await stop_bot(session_data["bot_id"])
                logger.info(f"✅ Stopped bot: {session_data['bot_id']}")
            except Exception as e:
                logger.warning(f"Failed to stop bot: {e}")
        
        # Delete Daily room if it exists
        if "daily_room" in session_data:
            try:
                from daily_service import get_daily_service
                daily_service = get_daily_service()
                if daily_service:
                    room_name = session_data["daily_room"]["name"]
                    await daily_service.delete_room(room_name)
                    logger.info(f"✅ Deleted Daily room: {room_name}")
            except Exception as e:
                logger.warning(f"Failed to delete Daily room: {e}")
        
        logger.info(f"Stream ended for session {session_id[:8]}")
        return {"message": "Stream ended successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error ending stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a browser session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        browser = session_data["browser"]
        
        # Delete Daily room if exists
        if "daily_room" in session_data:
            try:
                from daily_service import get_daily_service
                daily_service = get_daily_service()
                if daily_service:
                    room_name = session_data["daily_room"]["name"]
                    await daily_service.delete_room(room_name)
            except Exception as e:
                logger.warning(f"Failed to delete Daily room: {e}")
        
        # Close browser
        await browser.close()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Session {session_id[:8]} closed")
        return {"message": f"Session {session_id} closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all browser sessions on shutdown"""
    logger.info("Shutting down, closing all browser sessions...")
    for session_id, data in list(active_sessions.items()):
        try:
            await data["browser"].close()
        except:
            pass
    active_sessions.clear()


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Browser-Use FastAPI Agent")
    print("="*60)
    print("\nStarting server on http://localhost:8080")
    print("API docs available at http://localhost:8080/docs")
    print("\nMake sure OPENAI_API_KEY is set in your environment!")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
