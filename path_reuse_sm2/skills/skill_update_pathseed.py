from yasmin.blackboard import Blackboard
from path_reuse_sm2.core.base import Skill
from path_reuse_sm2.core.plugin import skill

@skill(names=["SkillUpdatePathSeed", "UpdatePathSeed"])
class SkillUpdatePathSeed(Skill):
    def run(self, *, bb: Blackboard, **kwargs) -> bool:
        self.node.get_logger().info("[SkillUpdatePathSeed] Update (dummy)")
        return True
# path_reuse_sm2/skills/skill_update_pathseed.py
from path_reuse_sm2.core.state import SkillState
from path_reuse_sm2.core.plugin import skill

@skill("SkillUpdatePathSeed")
class SkillUpdatePathSeed(SkillState):
    def __init__(self, node, **kwargs):
        super().__init__(node, name="SkillUpdatePathSeed")

    def run(self, bb, **kwargs) -> bool:
        self.node.get_logger().info("[SkillUpdatePathSeed] Update (dummy)")
        return True
