from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with OpenAI's GPT API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **Course Content Search Tool**: For questions about specific course content, lessons, or detailed educational materials
2. **Course Outline Tool**: For questions about course structure, lesson lists, course overviews, or complete course outlines

Tool Usage Guidelines:
- **Course outline/structure questions**: Use the course outline tool to get the complete course structure with all lessons
- **Specific content questions**: Use the content search tool for detailed information within courses
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use the outline tool, then provide the complete course title, course link, and full lesson list with numbers and titles
- **Course content questions**: Use the content search tool, then answer based on results
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

For outline queries, ensure you return:
- Course title (with clickable link if available)
- Course instructor
- Complete numbered list of all lessons with titles (and clickable links if available)

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build messages with system prompt and conversation history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        if conversation_history:
            messages.append({"role": "system", "content": f"Previous conversation:\n{conversation_history}"})
        
        messages.append({"role": "user", "content": query})
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        # Get response from OpenAI
        response = self.client.chat.completions.create(**api_params)
        
        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({
            "role": "assistant", 
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls
        })
        
        # Execute all tool calls and collect results
        for tool_call in initial_response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                import json
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, 
                    **json.loads(tool_call.function.arguments)
                )
                
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.id
                })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content