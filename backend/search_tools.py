from typing import Dict, Any, Optional, Protocol, List
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header with invisible clickable link
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
            
            if lesson_link:
                # Create clickable link that opens in new tab
                header = f"[{course_title}"
                if lesson_num is not None:
                    header += f" - Lesson {lesson_num}"
                header += f"]({lesson_link})"
            else:
                header = f"[{course_title}"
                if lesson_num is not None:
                    header += f" - Lesson {lesson_num}"
                header += "]"
            
            # Track source for the UI with clickable links
            if lesson_link:
                source = f"[{course_title}"
                if lesson_num is not None:
                    source += f" - Lesson {lesson_num}"
                source += f"]({lesson_link})"
            else:
                source = course_title
                if lesson_num is not None:
                    source += f" - Lesson {lesson_num}"
            sources.append(source)
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines and lesson information"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get the complete outline of a specific course including all lessons",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_name": {
                            "type": "string",
                            "description": "Course title or partial course name to get the outline for"
                        }
                    },
                    "required": ["course_name"]
                }
            }
        }
    
    def execute(self, course_name: str) -> str:
        """
        Execute the course outline tool with given course name.
        
        Args:
            course_name: Course name to get outline for
            
        Returns:
            Formatted course outline with lessons or error message
        """
        # First, resolve the course name to get exact match
        exact_course_title = self.store._resolve_course_name(course_name)
        if not exact_course_title:
            return f"No course found matching '{course_name}'"
        
        # Get the course metadata
        try:
            results = self.store.course_catalog.get(ids=[exact_course_title])
            if not results or not results.get('metadatas'):
                return f"Course metadata not found for '{exact_course_title}'"
            
            metadata = results['metadatas'][0]
            course_title = metadata.get('title', exact_course_title)
            instructor = metadata.get('instructor', 'Unknown')
            course_link = metadata.get('course_link')
            lessons_json = metadata.get('lessons_json')
            
            # Parse lessons data
            lessons = []
            if lessons_json:
                import json
                lessons = json.loads(lessons_json)
            
            # Format the course outline
            return self._format_course_outline(course_title, instructor, course_link, lessons)
            
        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"
    
    def _format_course_outline(self, title: str, instructor: str, course_link: Optional[str], lessons: List[Dict[str, Any]]) -> str:
        """Format course outline with enhanced visual presentation"""
        formatted = []
        
        # Enhanced course header with visual separators and improved styling
        formatted.append("---")
        formatted.append("")
        
        # Course title with enhanced visual presentation
        if course_link:
            formatted.append(f"# 🎓 **[{title}]({course_link})**")
        else:
            formatted.append(f"# 🎓 **{title}**")
        
        formatted.append("")
        
        # Instructor with enhanced styling
        if instructor:
            formatted.append(f"👨‍🏫 **Instructor:** {instructor}")
            formatted.append("")
        
        # Course link section (if available)
        if course_link:
            formatted.append(f"🔗 **Course Link:** [Access Course]({course_link})")
            formatted.append("")
        
        formatted.append("---")
        formatted.append("")
        
        # Enhanced lessons section
        if lessons:
            formatted.append("## 📚 **Course Lessons**")
            formatted.append("")
            
            # Sort lessons by lesson number
            sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 0))
            
            for lesson in sorted_lessons:
                lesson_num = lesson.get('lesson_number')
                lesson_title = lesson.get('lesson_title', 'Untitled')
                lesson_link = lesson.get('lesson_link')
                
                if lesson_link:
                    formatted.append(f"**{lesson_num}.** 📖 **[{lesson_title}]({lesson_link})**")
                else:
                    formatted.append(f"**{lesson_num}.** 📖 **{lesson_title}**")
        else:
            formatted.append("## 📚 **Course Lessons**")
            formatted.append("")
            formatted.append("ℹ️ No lessons available for this course.")
        
        formatted.append("")
        formatted.append("---")
        
        return "\n".join(formatted)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        # Handle OpenAI function format
        if "function" in tool_def:
            tool_name = tool_def["function"].get("name")
        else:
            tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []