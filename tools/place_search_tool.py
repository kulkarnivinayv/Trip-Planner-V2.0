import os
from utils.place_info_search import GooglePlaceSearchTool, TavilyPlaceSearchTool
from typing import List
from langchain.tools import tool
from dotenv import load_dotenv

class PlaceSearchTool:
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.environ.get("GPLACES_API_KEY")
        self.google_places_search = GooglePlaceSearchTool(self.google_api_key)
        self.tavily_search = TavilyPlaceSearchTool()
        self.place_search_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the place search tool"""
        @tool
        def search_attractions(city: str) -> str:
            """Search attractions of a city. Always call this with the argument name 'city'."""
            try:
                attraction_result = self.google_places_search.google_search_attractions(city)
                if attraction_result:
                    return f"Following are the attractions of {city} as suggested by google: {attraction_result}"
            except Exception as e:
                tavily_result = self.tavily_search.tavily_search_attractions(city)
                return f"Google cannot find the details due to {e}. \nFollowing are the attractions of {city}: {tavily_result}"  ## Fallback search using tavily in case google places fail
        
        @tool
        def search_restaurants(city: str) -> str:
            """Search restaurants of a city. Always call this with the argument name 'city'."""
            try:
                restaurants_result = self.google_places_search.google_search_restaurants(city)
                if restaurants_result:
                    return f"Following are the restaurants of {city} as suggested by google: {restaurants_result}"
            except Exception as e:
                tavily_result = self.tavily_search.tavily_search_restaurants(city)
                return f"Google cannot find the details due to {e}. \nFollowing are the restaurants of {city}: {tavily_result}"  ## Fallback search using tavily in case google places fail
        
        @tool
        def search_activities(city: str) -> str:
            """Search activities of a city. Always call this with the argument name 'city'."""
            try:
                restaurants_result = self.google_places_search.google_search_activity(city)
                if restaurants_result:
                    return f"Following are the activities in and around {city} as suggested by google: {restaurants_result}"
            except Exception as e:
                tavily_result = self.tavily_search.tavily_search_activity(city)
                return f"Google cannot find the details due to {e}. \nFollowing are the activities of {city}: {tavily_result}"  ## Fallback search using tavily in case google places fail
        
        @tool
        def search_transportation(city: str) -> str:
            """Search transportation of a city. Always call this with the argument name 'city'."""
            try:
                restaurants_result = self.google_places_search.google_search_transportation(city)
                if restaurants_result:
                    return f"Following are the modes of transportation available in {city} as suggested by google: {restaurants_result}"
            except Exception as e:
                tavily_result = self.tavily_search.tavily_search_transportation(city)
                return f"Google cannot find the details due to {e}. \nFollowing are the modes of transportation available in {city}: {tavily_result}"  ## Fallback search using tavily in case google places fail
        
        return [search_attractions, search_restaurants, search_activities, search_transportation]