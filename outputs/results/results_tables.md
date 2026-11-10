# Test results (mean [95% CI])

### 10% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.95 [0.9, 0.99] | 0.1 [0.05, 0.16] | 7.94 [6.88, 9.01] | 8.73 [7.56, 10] | 0.94 [0.89, 0.98] | 8.95 [7.79, 10.3] | 4.73 [4.27, 5.21] | 2.75 [2.35, 3.2] | 0.314 [0.279, 0.351] | 0.0312 [0.0253, 0.0377] | 439 [439, 439] |
| Isolation Forest alone | 0.71 [0.62, 0.8] | 0 [0, 0] | 10.3 [8.81, 11.6] | 16.6 [14.2, 19] | 0.71 [0.62, 0.8] | 16.6 [14.2, 19] | 8.73 [8.19, 9.31] | 0.284 [0.119, 0.45] | 0.755 [0.648, 0.869] | 0.000909 [0.00039, 0.00149] | 141 [140, 141] |

### 30% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.89 [0.83, 0.95] | 0.07 [0.02, 0.12] | 9.36 [8.15, 10.6] | 11.9 [10, 13.9] | 0.89 [0.83, 0.95] | 11.9 [10.1, 13.9] | 4.41 [3.96, 4.87] | 2.3 [1.85, 2.75] | 0.342 [0.299, 0.394] | 0.0199 [0.0152, 0.0247] | 439 [439, 439] |
| Isolation Forest alone | 0.76 [0.67, 0.84] | 0.04 [0.01, 0.08] | 10.9 [8.97, 12.8] | 15.5 [13.2, 17.9] | 0.75 [0.66, 0.83] | 15.8 [13.5, 18.2] | 8.25 [7.8, 8.72] | 1.61 [1.21, 2.09] | 0.39 [0.325, 0.466] | 0.0063 [0.00461, 0.00831] | 135 [135, 135] |

### 50% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.91 [0.85, 0.96] | 0.04 [0.01, 0.08] | 11.2 [9.71, 12.8] | 12.5 [10.9, 14.2] | 0.91 [0.85, 0.96] | 12.5 [10.9, 14.2] | 3.92 [3.53, 4.33] | 1.73 [1.4, 2.09] | 0.416 [0.368, 0.474] | 0.0121 [0.00903, 0.0153] | 439 [439, 439] |
| Isolation Forest alone | 0.83 [0.75, 0.9] | 0.23 [0.15, 0.31] | 10.2 [8.55, 11.9] | 13.6 [11.5, 15.8] | 0.82 [0.74, 0.89] | 14.3 [12.2, 16.6] | 8.22 [7.85, 8.63] | 4.46 [3.93, 5] | 0.204 [0.179, 0.231] | 0.0225 [0.0188, 0.0266] | 130 [130, 130] |

### All scenarios

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.917 [0.883, 0.947] | 0.07 [0.0433, 0.1] | 9.47 [8.7, 10.2] | 11 [10.1, 12] | 0.913 [0.88, 0.943] | 11.1 [10.2, 12.1] | 4.35 [4.1, 4.61] | 2.26 [2.02, 2.49] | 0.352 [0.327, 0.377] | 0.0211 [0.0181, 0.0242] | 439 [439, 439] |
| Isolation Forest alone | 0.767 [0.717, 0.81] | 0.09 [0.0567, 0.123] | 10.4 [9.45, 11.4] | 15.2 [13.9, 16.6] | 0.76 [0.71, 0.807] | 15.6 [14.3, 16.9] | 8.37 [8.11, 8.67] | 2.12 [1.82, 2.43] | 0.331 [0.299, 0.365] | 0.00989 [0.00827, 0.0117] | 135 [135, 136] |