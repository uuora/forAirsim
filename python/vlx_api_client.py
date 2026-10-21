"""VLX-Seek API client for Om AI Lab cloud inference service.

This adapter replaces the local VLXSeekWorker with cloud API calls.
"""
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image


class VLXSeekAPIClient:
    """Cloud API client compatible with VLXSeekWorker interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        model: str = "vlx-seek-1.5-3b",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """Initialize VLX-Seek API client.

        Args:
            api_key: Om AI Lab API key. Falls back to OM_API_KEY env var.
            api_endpoint: API endpoint URL. Falls back to OM_API_ENDPOINT env var.
            model: Model ID, either "vlx-seek" or "vlx-seek-1.5-3b".
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
        """
        self.api_key = api_key or os.environ.get("OM_API_KEY")
        self.api_endpoint = api_endpoint or os.environ.get("OM_API_ENDPOINT")
        self.model = model
        # The current cloud endpoint request contains image and text only.
        # Keep this explicit until a documented external-proposal field exists.
        self.request_mode = "image_text"
        self.proposals_transmitted = False
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("API key required: pass api_key or set OM_API_KEY env var")
        if not self.api_endpoint:
            raise ValueError("API endpoint required: pass api_endpoint or set OM_API_ENDPOINT")

        # Remove trailing slash from endpoint
        self.api_endpoint = self.api_endpoint.rstrip("/")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # Track usage for reporting
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_took_ms = 0

    def upload_image(self, image: Image.Image, format: str = "JPEG") -> str:
        """Upload image and return a temporary URL.

        For now, encode as base64 data URL since the API doc doesn't specify
        an image upload endpoint. If Om AI Lab provides a dedicated upload
        endpoint later, this method should be updated.

        Args:
            image: PIL Image to upload.
            format: Image format (JPEG or PNG).

        Returns:
            Image URL (currently a data URL).
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{encoded}"

    def _build_prompt_for_grounding(
        self,
        description: str,
        bbox_count: int,
        lang: str = "en"
    ) -> str:
        """Build prompt for grounding task based on model and template.

        Args:
            description: Target description (e.g., "the rightmost red balloon").
            bbox_count: Number of bounding boxes provided.
            lang: Language code.

        Returns:
            Formatted prompt string.
        """
        # For vlx-seek-1.5-3b: use the detection template
        if self.model == "vlx-seek-1.5-3b":
            # Simple detection template: "识别每个：{{target}}"
            if lang == "zh":
                return f"识别每个：{description}。"
            else:
                # English approximation
                return f"Identify each: {description}."

        # For original vlx-seek: use the detection template
        elif self.model == "vlx-seek":
            if lang == "zh":
                return f"请在图像中检测 {description}，并用对象索引回答这个问题。"
            else:
                return f"Please detect {description} in the image and answer with object indices."

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def ground(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        description: str,
        lang: str = "en",
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        """Ground description to one of the provided bounding boxes.

        This method calls the cloud image+text API. ``boxes`` are accepted for
        interface compatibility but are not included in the HTTP payload.

        Args:
            image: PIL Image in RGB mode.
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...].
            description: Target description text.
            lang: Language code ("en" or "zh").
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments (ignored for API compatibility).

        Returns:
            Dictionary with "answer" and "result_bbox_list" keys, matching
            the local VLXSeekWorker output format.
        """
        if len(boxes) > 100:
            raise ValueError(f"VLX-Seek supports at most 100 proposals, got {len(boxes)}")

        # Convert PIL image to URL (data URL for now)
        image_url = self.upload_image(image)

        # Build the prompt
        prompt = self._build_prompt_for_grounding(description, len(boxes), lang)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0
        }

        # External proposals are not supported by this request implementation.

        # Make API request with retries
        url = f"{self.api_endpoint}/os/v2/chat/completions"

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                # Track usage
                self.total_requests += 1
                if "took" in result:
                    self.total_took_ms += result["took"]
                if "usage" in result:
                    self.total_prompt_tokens += result["usage"].get("prompt_tokens", 0)
                    self.total_completion_tokens += result["usage"].get("completion_tokens", 0)

                # Convert API response to VLXSeekWorker format
                return self._convert_response(result, boxes)

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API request failed after {self.max_retries} attempts: {e}")

    def _convert_response(
        self,
        api_response: Dict[str, Any],
        input_boxes: List[List[float]]
    ) -> Dict[str, Any]:
        """Convert API response to VLXSeekWorker output format.

        Args:
            api_response: Raw API response.
            input_boxes: Original input bounding boxes.

        Returns:
            Dictionary with "answer" and "result_bbox_list" keys.
        """
        # Check for error in the response
        # Note: Successful responses may not have a 'code' field
        if "error" in api_response or ("code" in api_response and api_response["code"] != 200):
            error_msg = api_response.get("error") or api_response.get("message", "Unknown API error")
            raise RuntimeError(f"API returned error (code={api_response.get('code')}): {error_msg}")

        choices = api_response.get("choices", [])
        if not choices:
            raise RuntimeError("API returned no choices")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        result_bbox_list = choice.get("result_bbox_list", [])

        # Preserve the service's returned coordinates. The cloud request does
        # not contain input_boxes, so proximity cannot establish proposal use.
        mapped_boxes = []
        for bbox in result_bbox_list:
            mapped_boxes.append({
                "object_index": bbox.get("object_index", ""),
                "xmin": bbox["xmin"],
                "ymin": bbox["ymin"],
                "xmax": bbox["xmax"],
                "ymax": bbox["ymax"],
                "label": bbox.get("label", "")
            })

        return {
            "answer": content,
            "result_bbox_list": mapped_boxes
        }

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get accumulated usage statistics.

        Returns:
            Dictionary with request count, token usage, and timing.
        """
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_took_ms": self.total_took_ms,
            "average_took_ms": (self.total_took_ms / self.total_requests
                                if self.total_requests > 0 else 0)
        }


def create_api_worker(model: str = "vlx-seek-1.5-3b", **kwargs) -> VLXSeekAPIClient:
    """Factory function matching the load_worker pattern in vlx_seek_probe.py.

    Args:
        model: Model ID.
        **kwargs: Additional arguments for VLXSeekAPIClient.

    Returns:
        Initialized API client.
    """
    return VLXSeekAPIClient(model=model, **kwargs)
