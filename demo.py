from core.optimizer import ScriptOptimizer

script = (
    "Imagine opening GTA 6. Rockstar officially revealed new screenshots. "
    "But fans completely missed one detail!"
)

opt = ScriptOptimizer()
print(opt.optimize(script, style="documentary"))
