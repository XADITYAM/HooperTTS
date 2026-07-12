from core.optimizer import ScriptOptimizer

o=ScriptOptimizer()
out=o.optimize("Imagine opening GTA 6. But Rockstar officially confirmed it.")
assert "Imagine" in out
assert "OFFICIALLY" in out
print("PASS")
