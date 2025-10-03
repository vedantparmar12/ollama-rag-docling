"""Real-Time RAG for Dynamic Data Sources

Enables RAG queries over dynamic, real-time data sources like:
- Web APIs (weather, stock prices, news)
- Databases (current inventory, user data)
- Live feeds (social media, IoT sensors)

This extends traditional RAG by combining static document retrieval
with fresh, up-to-date information from external sources.
"""

from typing import Dict, Any, List, Optional, Callable
import asyncio
import httpx
from datetime import datetime
import json


class RealtimeDataSource:
    """Base class for real-time data sources."""

    def __init__(self, name: str, cache_ttl: int = 60):
        """
        Initialize data source.

        Args:
            name: Source name
            cache_ttl: Cache time-to-live in seconds
        """
        self.name = name
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def fetch(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from source.

        Args:
            query: User query
            params: Extracted parameters

        Returns:
            Fetched data
        """
        raise NotImplementedError("Subclasses must implement fetch()")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False

        timestamp = self._cache_timestamps[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Dict[str, Any]):
        """Set cache data."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()


class WeatherDataSource(RealtimeDataSource):
    """Real-time weather data source."""

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 600):
        super().__init__("weather", cache_ttl)
        self.api_key = api_key

    async def fetch(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch current weather data."""
        location = params.get('location', 'unknown')
        cache_key = f"weather_{location}"

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # For demo purposes, return mock data
        # In production, integrate with OpenWeatherMap, Weather API, etc.
        data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "timestamp": datetime.now().isoformat(),
            "source": "weather_api"
        }

        self._set_cache(cache_key, data)
        return data


class StockDataSource(RealtimeDataSource):
    """Real-time stock price data source."""

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 30):
        super().__init__("stock", cache_ttl)
        self.api_key = api_key

    async def fetch(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch current stock price."""
        symbol = params.get('symbol', 'UNKNOWN').upper()
        cache_key = f"stock_{symbol}"

        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Mock data (integrate with Alpha Vantage, Yahoo Finance, etc.)
        data = {
            "symbol": symbol,
            "price": "$150.25",
            "change": "+2.5%",
            "volume": "1.2M",
            "timestamp": datetime.now().isoformat(),
            "source": "stock_api"
        }

        self._set_cache(cache_key, data)
        return data


class DatabaseDataSource(RealtimeDataSource):
    """Real-time database query source."""

    def __init__(self, connection_string: Optional[str] = None, cache_ttl: int = 60):
        super().__init__("database", cache_ttl)
        self.connection_string = connection_string

    async def fetch(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from database."""
        table = params.get('table', 'unknown')
        cache_key = f"db_{table}_{hash(str(params))}"

        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Mock data (integrate with SQLAlchemy, asyncpg, etc.)
        data = {
            "table": table,
            "results": [
                {"id": 1, "value": "Sample Data 1"},
                {"id": 2, "value": "Sample Data 2"}
            ],
            "count": 2,
            "timestamp": datetime.now().isoformat(),
            "source": "database"
        }

        self._set_cache(cache_key, data)
        return data


class RealtimeRetriever:
    """
    Hybrid retriever combining static RAG with real-time data.

    Workflow:
    1. Detect if query needs real-time data
    2. Extract parameters from query
    3. Fetch from appropriate data source(s)
    4. Combine with static RAG results
    5. Generate unified response
    """

    def __init__(self, static_retriever=None):
        """
        Initialize real-time retriever.

        Args:
            static_retriever: Traditional RAG retriever for static docs
        """
        self.static_retriever = static_retriever
        self.data_sources: Dict[str, RealtimeDataSource] = {}

        # Register default data sources
        self.register_source(WeatherDataSource())
        self.register_source(StockDataSource())
        self.register_source(DatabaseDataSource())

    def register_source(self, source: RealtimeDataSource):
        """Register a new data source."""
        self.data_sources[source.name] = source
        print(f"✅ Registered real-time source: {source.name}")

    def detect_realtime_intent(self, query: str) -> Optional[str]:
        """
        Detect if query requires real-time data.

        Args:
            query: User query

        Returns:
            Data source name if real-time query, None otherwise
        """
        query_lower = query.lower()

        # Weather queries
        weather_keywords = ['weather', 'temperature', 'forecast', 'raining', 'sunny']
        if any(keyword in query_lower for keyword in weather_keywords):
            if any(word in query_lower for word in ['current', 'now', 'today', 'what is', 'what\'s']):
                return 'weather'

        # Stock queries
        stock_keywords = ['stock', 'price', 'shares', 'trading', 'market']
        if any(keyword in query_lower for keyword in stock_keywords):
            if any(word in query_lower for word in ['current', 'now', 'today', 'latest']):
                return 'stock'

        # Database queries
        db_keywords = ['inventory', 'available', 'in stock', 'count', 'how many']
        if any(keyword in query_lower for keyword in db_keywords):
            return 'database'

        return None

    def extract_parameters(self, query: str, source_type: str) -> Dict[str, Any]:
        """
        Extract parameters from query for data source.

        Args:
            query: User query
            source_type: Type of data source

        Returns:
            Extracted parameters
        """
        params = {}

        if source_type == 'weather':
            # Extract location
            # Simple heuristic: look for "in [location]" or "at [location]"
            import re
            location_match = re.search(r'in ([A-Z][a-z]+)', query)
            if not location_match:
                location_match = re.search(r'at ([A-Z][a-z]+)', query)
            if location_match:
                params['location'] = location_match.group(1)
            else:
                params['location'] = 'New York'  # Default

        elif source_type == 'stock':
            # Extract stock symbol
            import re
            # Look for uppercase ticker symbols
            symbol_match = re.search(r'\b([A-Z]{2,5})\b', query)
            if symbol_match:
                params['symbol'] = symbol_match.group(1)
            else:
                params['symbol'] = 'AAPL'  # Default

        elif source_type == 'database':
            # Extract table/entity name
            params['table'] = 'products'  # Default

        return params

    async def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve information combining real-time and static sources.

        Args:
            query: User query
            top_k: Number of static results

        Returns:
            Combined retrieval results
        """
        results = {
            'query': query,
            'realtime_data': None,
            'static_data': None,
            'combined': True,
            'timestamp': datetime.now().isoformat()
        }

        # Detect real-time intent
        source_type = self.detect_realtime_intent(query)

        # Fetch real-time data if needed
        if source_type and source_type in self.data_sources:
            print(f"🔴 Real-time query detected: {source_type}")
            params = self.extract_parameters(query, source_type)
            source = self.data_sources[source_type]

            try:
                realtime_data = await source.fetch(query, params)
                results['realtime_data'] = realtime_data
                print(f"✅ Real-time data fetched from {source_type}")
            except Exception as e:
                print(f"⚠️ Failed to fetch real-time data: {e}")
                results['realtime_error'] = str(e)

        # Fetch static data if retriever available
        if self.static_retriever:
            try:
                static_results = await self._fetch_static_data(query, top_k)
                results['static_data'] = static_results
                print(f"✅ Static data retrieved ({len(static_results)} results)")
            except Exception as e:
                print(f"⚠️ Failed to fetch static data: {e}")
                results['static_error'] = str(e)

        return results

    async def _fetch_static_data(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fetch data from static RAG retriever."""
        # If retriever has async retrieve method
        if hasattr(self.static_retriever, 'retrieve'):
            if asyncio.iscoroutinefunction(self.static_retriever.retrieve):
                return await self.static_retriever.retrieve(query, top_k=top_k)
            else:
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.static_retriever.retrieve(query, top_k=top_k)
                )
        else:
            return []

    def format_combined_context(self, results: Dict[str, Any]) -> str:
        """
        Format combined real-time and static results for LLM context.

        Args:
            results: Combined retrieval results

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add real-time data
        if results.get('realtime_data'):
            rt_data = results['realtime_data']
            context_parts.append("**Real-Time Information:**")
            context_parts.append(json.dumps(rt_data, indent=2))
            context_parts.append("")

        # Add static data
        if results.get('static_data'):
            context_parts.append("**Relevant Documents:**")
            for i, doc in enumerate(results['static_data'][:5], 1):
                context_parts.append(f"{i}. {doc.get('text', '')[:200]}...")
            context_parts.append("")

        return "\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    async def test_realtime_retriever():
        retriever = RealtimeRetriever()

        # Test queries
        test_queries = [
            "What's the current weather in Paris?",
            "What is the current stock price of AAPL?",
            "How many products are in stock?",
            "Tell me about machine learning"  # Static query
        ]

        print("\n" + "="*60)
        print("REAL-TIME RAG RETRIEVER TEST")
        print("="*60)

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            results = await retriever.retrieve(query)

            if results.get('realtime_data'):
                print(f"🔴 Real-Time Data: {results['realtime_data']}")
            else:
                print(f"📄 Static query (no real-time data)")

            print("-" * 60)

    # Run test
    asyncio.run(test_realtime_retriever())
    print("\n✅ Real-time RAG test completed")
