# IGPO Tool Server - Web Search

Lightweight web search tool server for IGPO training.

## Quick Start

1. Get a Serper API key from https://serper.dev/ (2,500 free searches)

2. Edit `config.yaml`:
```yaml
serper_api_key: "your_api_key_here"
```

3. Run training - the tool server will be used automatically.

## Configuration

```yaml
# config.yaml
search_engine: "google"     # or "bing"
search_top_k: 10            # results per query
serper_api_key: "xxx"       # Serper API key
```

## Supported Search Engines

| Engine | Provider | Free Tier |
|--------|----------|-----------|
| Google | Serper API | 2,500 searches |
| Bing | Azure | Pay as you go |

## Files

```
tools_server/
├── config.yaml      # Configuration
├── handler.py       # Web search handler
├── tools.py         # Tool definition
├── util.py          # MessageClient
└── search/
    └── search_api.py  # Search implementations
```
