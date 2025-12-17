"""
Browser Streaming Bot for Daily.co

This bot joins a Daily room and streams browser screenshots as video frames.
Based on Pipecat's architecture for real-time video streaming.
"""

import asyncio
import base64
import io
import logging
import os
from typing import Optional

from PIL import Image
from pipecat.frames.frames import Frame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger = logging.getLogger(__name__)


class BrowserFrameStreamer(FrameProcessor):
    """
    Captures frames from the browser and streams them to Daily.
    """
    
    def __init__(self, session_id: str, fastapi_url: str = "https://chromeengine-739298578243.us-central1.run.app"):
        super().__init__()
        self.session_id = session_id
        self.fastapi_url = fastapi_url
        self._is_streaming = True
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames in the pipeline."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
    
    async def capture_and_stream(self):
        """Continuously capture browser screenshots and push as video frames."""
        import aiohttp
        
        frame_count = 0
        logger.info("🎬 Starting frame capture loop...")
        
        while self._is_streaming:
            try:
                # Get screenshot from FastAPI
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.fastapi_url}/streaming/get-screenshot/{self.session_id}",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            screenshot_b64 = data.get('screenshot')
                            
                            if screenshot_b64:
                                # Decode base64 to image
                                img_data = base64.b64decode(screenshot_b64)
                                img = Image.open(io.BytesIO(img_data))
                                
                                # Get actual image dimensions (preserve aspect ratio)
                                width, height = img.size
                                
                                # Convert to RGB if needed (remove alpha channel)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Convert to raw RGB bytes - let Daily handle aspect ratio
                                rgb_bytes = img.tobytes()
                                expected_bytes = width * height * 3
                                
                                if len(rgb_bytes) != expected_bytes:
                                    logger.error(f"❌ Byte size mismatch! Got {len(rgb_bytes)}, expected {expected_bytes} for {width}x{height}")
                                    await asyncio.sleep(0.2)
                                    continue
                                
                                # Create frame with actual image dimensions
                                frame = OutputImageRawFrame(
                                    image=rgb_bytes,
                                    size=(width, height),
                                    format="RGB"
                                )
                                
                                # Push frame to pipeline - let Daily handle it
                                try:
                                    await self.push_frame(frame)
                                    frame_count += 1
                                    if frame_count % 15 == 0:  # Log every 15 frames (~3 seconds at 5 FPS)
                                        logger.info(f"📸 Sent {frame_count} frames, latest: {width}x{height} RGB ({len(rgb_bytes)} bytes)")
                                except Exception as frame_error:
                                    logger.error(f"❌ Error pushing frame #{frame_count}: {frame_error}", exc_info=True)
                            else:
                                logger.warning("⚠️ No screenshot data in response")
                        elif response.status == 429:
                            # Rate limited - back off
                            logger.warning("⚠️ Rate limited (429), backing off for 2 seconds...")
                            await asyncio.sleep(2.0)
                        else:
                            logger.warning(f"⚠️ Screenshot request failed: HTTP {response.status}")
                        
            except asyncio.TimeoutError:
                logger.warning("⏱️ Screenshot request timed out")
            except Exception as e:
                logger.error(f"❌ Error capturing frame: {e}", exc_info=True)
            
            # Capture at ~5 FPS (every 200ms) - reduces rate limiting issues
            await asyncio.sleep(0.2)
        
        logger.info(f"🛑 Frame capture stopped. Total frames sent: {frame_count}")
    
    def stop_streaming(self):
        """Stop the streaming loop."""
        self._is_streaming = False


async def run_browser_bot(
    room_url: str,
    session_id: str,
    token: Optional[str] = None,
    fastapi_url: str = "https://chromeengine-739298578243.us-central1.run.app"
):
    """
    Run the browser streaming bot.
    
    Args:
        room_url: Daily.co room URL
        session_id: Browser session ID to stream
        token: Optional Daily meeting token
        fastapi_url: FastAPI server URL
    """
    
    # Create Daily transport
    transport = DailyTransport(
        room_url,
        token,
        "Browser Stream Bot",
        params=DailyParams(
            audio_in_enabled=False,   # No audio input
            audio_out_enabled=False,  # No audio output
            video_out_enabled=True,   # Video output enabled!
            video_out_is_live=True,   # CRITICAL: Enable live video streaming (not still frame)
            camera_out_enabled=True,   # Explicitly enable camera output
            video_out_width=1280,      # Hint for Daily (browser window size)
            video_out_height=720,      # Hint for Daily (browser window size)
            video_out_framerate=5,     # Match our capture rate (5 FPS - reduces rate limiting)
            video_out_color_format="RGB",  # RGB format
            vad_enabled=False,        # No voice activity detection needed
            transcription_enabled=False,
        ),
    )
    
    # Create frame streamer
    streamer = BrowserFrameStreamer(session_id, fastapi_url)
    
    # Create pipeline: streamer -> transport output
    pipeline = Pipeline([
        streamer,
        transport.output(),
    ])
    
    # Create task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )
    
    # Track if streaming has started (using a list to work around closure)
    streaming_started = [False]
    
    # Event handlers
    async def start_streaming_safely():
        """Start streaming after a delay - Daily will handle camera initialization."""
        if streaming_started[0]:
            return
        
        logger.info("⏳ Waiting for transport to be ready...")
        await asyncio.sleep(3.0)  # Give Daily transport time to fully initialize
        
        logger.info("🎬 Starting video stream...")
        streaming_started[0] = True
        asyncio.create_task(streamer.capture_and_stream())
    
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"👋 First participant joined: {participant}")
        # Start streaming after delay
        asyncio.create_task(start_streaming_safely())
    
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"👋 Participant left: {participant}")
        # If no participants left, we could stop the bot
        if len(transport.participants()) == 1:  # Only bot left
            logger.info("No more participants, stopping bot...")
            streamer.stop_streaming()
            await task.cancel()
    
    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        logger.info(f"📞 Call state: {state}")
        # Start streaming when fully joined (if not already started)
        if state == "joined" and not streaming_started[0]:
            asyncio.create_task(start_streaming_safely())
    
    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    
    logger.info(f"🤖 Starting browser streaming bot for session {session_id[:8]}")
    logger.info(f"🎥 Streaming to room: {room_url}")
    
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        streamer.stop_streaming()
        logger.info("🛑 Browser streaming bot stopped")


# Singleton bot manager
_active_bots: dict[str, asyncio.Task] = {}


async def start_bot_for_session(
    session_id: str,
    room_url: str,
    token: Optional[str] = None,
    fastapi_url: str = "https://chromeengine-739298578243.us-central1.run.app"
) -> str:
    """
    Start a bot for a browser session.
    
    Returns:
        bot_id: Unique ID for this bot instance
    """
    bot_id = f"bot-{session_id}"
    
    if bot_id in _active_bots:
        logger.warning(f"Bot already running for session {session_id[:8]}")
        return bot_id
    
    # Start bot in background
    task = asyncio.create_task(
        run_browser_bot(room_url, session_id, token, fastapi_url)
    )
    _active_bots[bot_id] = task
    
    logger.info(f"✅ Started bot {bot_id} for session {session_id[:8]}")
    return bot_id


async def stop_bot(bot_id: str):
    """Stop a running bot."""
    if bot_id in _active_bots:
        task = _active_bots[bot_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        del _active_bots[bot_id]
        logger.info(f"🛑 Stopped bot {bot_id}")


def get_active_bots() -> list[str]:
    """Get list of active bot IDs."""
    return list(_active_bots.keys())

