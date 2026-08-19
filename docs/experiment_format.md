```
Experiment
├── experiment_id
├── timestamp
├── git_commit
├── workload
├── hardware
├── software
├── model
├── engine
├── configuration
├── metrics
├── errors
└── raw_results
```

The key idea is:
```
result.json
     ↓
must tell you exactly
how that number was produced
```
Six months later you should be able to look at a result and reproduce it.