from core.optimizer import ScriptOptimizer

sample_script = """
Rockstar has officially released new GTA 6 screenshots.
Fans immediately noticed several hidden details.
Will Rockstar reveal even more soon?
"""

optimizer = ScriptOptimizer()

result = optimizer.optimize(sample_script)

print(result)
