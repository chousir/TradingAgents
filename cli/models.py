from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
