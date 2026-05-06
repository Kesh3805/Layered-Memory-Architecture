import re

with open('experiments/results/retrieval_fast_1778046555.md', 'r', encoding='utf-8') as f:
    new_findings = f.read()

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

pattern = re.compile(r'<!-- RETRIEVAL_FINDINGS_START -->.*?<!-- RETRIEVAL_FINDINGS_END -->', re.DOTALL)
replacement = f'<!-- RETRIEVAL_FINDINGS_START -->\n\n{new_findings}\n\n<!-- RETRIEVAL_FINDINGS_END -->'
new_readme = pattern.sub(replacement, readme)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(new_readme)
print("Updated README.md")
