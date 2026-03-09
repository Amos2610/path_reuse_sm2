from __future__ import annotations
from typing import Dict, List, Type, Iterable, Callable
import importlib, pkgutil
from .base import Skill

_SKILLS: Dict[str, Type[Skill]] = {}

def skill(name: str | None = None, names: Iterable[str] | None = None) -> Callable[[Type[Skill]], Type[Skill]]:
    """@skill('SkillGraspObj') or @skill(names=['SkillGraspObj','GraspObj'])"""
    def _wrap(cls: Type[Skill]) -> Type[Skill]:
        keys: List[str] = []
        if name: keys.append(name)
        if names: keys.extend(names)
        if not keys: keys.append(getattr(cls, "name", cls.__name__))
        for k in keys:
            _SKILLS[k] = cls
        return cls
    return _wrap

def discover(package: str = "path_reuse_sm2.skills") -> None:
    """skills/ 配下を import して @skill 登録を発火"""
    try:
        pkg = importlib.import_module(package)
    except Exception as e:
        print(f"[discover] failed to import package {package}: {e}")
        return
    if not hasattr(pkg, "__path__"):
        return
    for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            print(f"[discover] failed to import module {m.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def get(name: str) -> Type[Skill]:
    return _SKILLS[name]

def names() -> List[str]:
    return sorted(_SKILLS.keys())
