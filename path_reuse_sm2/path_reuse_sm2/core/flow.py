from typing import Any, Dict, List, Union
FlowEntry = Union[str, Dict[str, Any]]

def normalize(flow_param: List[FlowEntry]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in (flow_param or []):
        if isinstance(e, str):
            out.append({"name": e, "args": {}})
        elif isinstance(e, dict):
            n = e.get("name")
            if not n:
                continue
            a = e.get("args", {})
            out.append({"name": n, "args": a if isinstance(a, dict) else {}})
    return out
