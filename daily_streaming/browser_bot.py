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
    
    def __init__(self, session_id: str, fastapi_url: str = "http://localhost:8000"):
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
        
        while self._is_streaming:
            try:
                # Get screenshot from FastAPI
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.fastapi_url}/streaming/get-screenshot/{self.session_id}"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            screenshot_b64 = data.get('screenshot')
                            
                            if screenshot_b64:
                                # Decode base64 to image
                                img_data = base64.b64decode(screenshot_b64)
                                img = Image.open(io.BytesIO(img_data))
                                
                                # Convert to raw bytes for Pipecat
                                frame = OutputImageRawFrame(
                                    image=img.tobytes(),
                                    size=img.size,
                                    format=img.format or "PNG"
                                )
                                
                                # Push frame to pipeline
                                await self.push_frame(frame)
                                logger.debug(f"📸 Pushed frame: {img.size}")
                        
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
            
            # Capture at ~10 FPS
            await asyncio.sleep(0.1)
    
    def stop_streaming(self):
        """Stop the streaming loop."""
        self._is_streaming = False


async def run_browser_bot(
    room_url: str,
    session_id: str,
    token: Optional[str] = None,
    fastapi_url: str = "http://localhost:8000"
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
            video_out_width=1280,     # Match canvas size
            video_out_height=720,
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
    
    # Event handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"👋 First participant joined: {participant}")
        # Start streaming browser frames
        asyncio.create_task(streamer.capture_and_stream())
    
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
    fastapi_url: str = "http://localhost:8000"
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

