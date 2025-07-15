# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import asyncio
import aiohttp
import json

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class ListTool(BaseTool):
    """
    tool_schema:
      type: "function"
      function:
        name: "tools/list"
        description: "Shows the list of all available tools, with information about name and parameters."
        paramters:
          type: "object"
          properties: null
          required: []
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_list = [
            {
                "name": "tools/execute_fstar",
                "description": "A tool that executes the given fstar code.",
                "parameters": {
                    "code": {
                        "type": "string",
                        "description": "F* code to execute"
                    }
                },
                "required": ["code"]
            }
        ]
        self._instance_dict = {}


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # print("TOOL_CALL list")
        return json.dumps(self.tool_list, indent=2), 0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass

class FStarExecutionTool(BaseTool):
    """
    tool_schema:
      type: "function"
      function:
        name: "tools/execute_fstar"
        description: "A tool that executes the given fstar code."
        paramters:
          type: "object"
          properties:
            code:
              type: string
              description: "F* code to execute"
          required: ["code"]
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # print("TOOL CALL execute_fstar")
        url = os.environ.get("FSTAR_VERIFIER_SERVER_HOST", "http://localhost:8005") + "/check_problem_solution"
        # print(parameters.keys())

        code = parameters["code"]

        payload = {
            "solution": code,
            "problem_id": kwargs["tools_kwargs"]["example_name"]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    result_msg = "Verification Success: " + str(resp_json.get("return_code", -2) == 0 and resp_json.get("score") == 1.0) + "\n"
                    return result_msg + resp_json.get("messages", ""), 0, {}
        except Exception as e:
            # print(f"Request failed for example {extra_info.get('example_name')}: {e}")
            return f"Runtime error occurred.\n{e.__class__}: {e}", 0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
