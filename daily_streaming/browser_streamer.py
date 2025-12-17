"""
Browser Frame Capture and Streaming Service

This service captures frames from a Chrome browser via CDP and prepares them for streaming.
For actual Daily.co streaming, we'll use the Daily.co SDK on the client side to display the browser.
"""

import asyncio
import base64
import logging
from typing import Optional, Callable
import websockets
import json

logger = logging.getLogger(__name__)


class BrowserStreamer:
    """
    Captures frames from Chrome via CDP.
    
    Note: For this use case, we'll use a simpler approach:
    - The client (web UI) will connect to the Chrome CDP endpoint
    - The client will capture frames and display them in the Daily room
    - This avoids complex server-side video encoding
    """
    
    def __init__(self, cdp_url: str):
        """
        Initialize browser streamer.
        
        Args:
            cdp_url: Chrome DevTools Protocol WebSocket URL
        """
        self.cdp_url = cdp_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_streaming = False
        self.frame_callback: Optional[Callable] = None
        
    async def connect(self):
        """Connect to Chrome CDP."""
        try:
            self.ws = await websockets.connect(self.cdp_url)
            logger.info(f"✅ Connected to Chrome CDP: {self.cdp_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CDP: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Chrome CDP."""
        if self.ws:
            await self.ws.close()
            self.ws = None
            logger.info("Disconnected from Chrome CDP")
    
    async def send_command(self, method: str, params: Optional[dict] = None) -> dict:
        """
        Send a CDP command.
        
        Args:
            method: CDP method name
            params: Method parameters
            
        Returns:
            dict: Command result
        """
        if not self.ws:
            raise Exception("Not connected to CDP")
        
        command = {
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        await self.ws.send(json.dumps(command))
        response = await self.ws.recv()
        return json.loads(response)
    
    async def capture_screenshot(self, format: str = "png") -> Optional[str]:
        """
        Capture a screenshot from the browser.
        
        Args:
            format: Image format ('png' or 'jpeg')
            
        Returns:
            str: Base64-encoded screenshot data
        """
        try:
            result = await self.send_command(
                "Page.captureScreenshot",
                {"format": format, "quality": 90 if format == "jpeg" else None}
            )
            return result.get("result", {}).get("data")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    async def start_screencast(self, fps: int = 30, quality: int = 80):
        """
        Start screen casting (streaming frames).
        
        Args:
            fps: Frames per second
            quality: JPEG quality (0-100)
        """
        try:
            await self.send_command(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": quality,
                    "maxWidth": 1920,
                    "maxHeight": 1080,
                    "everyNthFrame": max(1, 30 // fps)  # Convert FPS to frame skip
                }
            )
            self.is_streaming = True
            logger.info(f"✅ Started screencast at {fps} FPS")
        except Exception as e:
            logger.error(f"Failed to start screencast: {e}")
            raise
    
    async def stop_screencast(self):
        """Stop screen casting."""
        try:
            await self.send_command("Page.stopScreencast")
            self.is_streaming = False
            logger.info("Stopped screencast")
        except Exception as e:
            logger.error(f"Failed to stop screencast: {e}")
    
    async def listen_for_frames(self, callback: Callable):
        """
        Listen for screencast frames.
        
        Args:
            callback: Function to call with each frame (receives base64 data)
        """
        if not self.ws:
            raise Exception("Not connected to CDP")
        
        self.frame_callback = callback
        
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                # Check for screencast frames
                if data.get("method") == "Page.screencastFrame":
                    params = data.get("params", {})
                    frame_data = params.get("data")
                    session_id = params.get("sessionId")
                    
                    # Acknowledge the frame
                    await self.send_command(
                        "Page.screencastFrameAck",
                        {"sessionId": session_id}
                    )
                    
                    # Call the callback with frame data
                    if frame_data and self.frame_callback:
                        await self.frame_callback(frame_data)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info("CDP connection closed")
        except Exception as e:
            logger.error(f"Error listening for frames: {e}")
        finally:
            self.is_streaming = False


# Global registry of active streamers by session ID
_active_streamers: dict[str, BrowserStreamer] = {}


def get_streamer(session_id: str) -> Optional[BrowserStreamer]:
    """Get the browser streamer for a session."""
    return _active_streamers.get(session_id)


def register_streamer(session_id: str, streamer: BrowserStreamer):
    """Register a browser streamer for a session."""
    _active_streamers[session_id] = streamer
    logger.info(f"Registered streamer for session {session_id[:8]}")


def unregister_streamer(session_id: str):
    """Unregister a browser streamer."""
    if session_id in _active_streamers:
        del _active_streamers[session_id]
        logger.info(f"Unregistered streamer for session {session_id[:8]}")

