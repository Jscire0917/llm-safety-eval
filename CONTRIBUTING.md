# Contributing to LLM-Safety-Eval

Thank you for considering contributing to LLM-Safety-Eval!  
This project is open to bug fixes, new features, documentation improvements, better tests, and more datasets/evaluators.

## How to Contribute

### 1. Fork & Clone

1. Fork the repository on GitHub.
2. Clone your fork locally:

'''bash
    git clone https://github.com/Jscire0917/llm-safety-eval.git
    cd llm-safety-eval
3. Set up the development environment
'''Bash
    python -m venv .venv
    source .venv/bin/activate   # or .venv\Scripts\activate on Windows
    pip install -e ".[test]"
4. Create Branch 
    git checkout -b feature/main
    feature/add-real-toxicity-dataset  
    fix/bias-score-zero-issue
    test/add-cost-tracker-tests
    docs/update-readme-quickstart
    chore/remove-unused-files

5. Make Changes:

Follow existing code style (PEP 8, black formatting recommended)
Add or update tests when you change behavior (pytest tests/)
Add comments for new or complex logic
Keep commits small and descriptive

'''Bash
    git add .
    git commit -m "Add RealToxicityPrompts subset and toxicity evaluation support"
6. Push & Open a Pull Request
Push your branch:
'''Bash
    git push origin feature/main

7. In the PR description:

Describe what you changed and why
Mention any related issues (Fixes #123)
Include before/after screenshots or test output if relevant

8. Development Guidelines:

Code style: Use black (optional: install pre-commit hooks)
Testing: All new features should have tests. Run pytest tests/ before pushing.
Commit messages: Use conventional style (e.g. "feat: add X", "fix: resolve Y", "docs: update Z")
Dependencies: Add new packages to pyproject.toml (not requirements.txt)
Environment: Use Python 3.10+ (as specified in pyproject.toml)

9. What to Expect After Submitting:

I (or maintainers) will review your PR as soon as possible.
We may ask for small changes or clarifications.
Once approved, your PR will be merged.

Thank You!
Every contribution — big or small — helps make this project better.
Bug reports, feature ideas, documentation fixes, and code improvements are all welcome.
Happy contributing!
— John
