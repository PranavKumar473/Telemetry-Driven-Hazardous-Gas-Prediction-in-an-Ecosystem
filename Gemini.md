
# Project Rules & Engineering Standards

1. **Incremental Execution:** Never modify multiple subsystems at once. Work strictly on the single sub-phase requested.
2. **Explain Simply (Learn-While-Building):** After every code change, generate a short plain-English walkthrough explaining *what* changed, *why* it was needed, and the exact input/output data shapes so I can defend every line in technical interviews.
3. **Prevent Data Leakage:** In time-series ML pipelines, always split data chronologically before fitting scalers or encoders. Never shuffle time-series windows randomly.
4. **Verify Before Moving On:** After editing any file, run the corresponding script or `pytest` check in the terminal and confirm zero errors before concluding the turn.
