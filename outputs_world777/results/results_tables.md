# Test results (mean [95% CI])

### 10% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.95 [0.9, 0.99] | 0.06 [0.02, 0.11] | 7.05 [6.05, 8.08] | 7.75 [6.61, 8.91] | 0.94 [0.88, 0.98] | 8.24 [6.95, 9.78] | 5.22 [4.82, 5.66] | 2.7 [2.3, 3.1] | 0.322 [0.289, 0.358] | 0.0332 [0.0266, 0.0403] | 577 [577, 577] |
| Isolation Forest alone | 0.72 [0.63, 0.8] | 0 [0, 0] | 10.2 [8.26, 12.3] | 16 [13.6, 18.6] | 0.72 [0.63, 0.8] | 16 [13.6, 18.6] | 7.67 [7.27, 8.08] | 0.0711 [0, 0.142] | 0.935 [0.872, 0.987] | 0.00026 [0, 0.000584] | 138 [138, 138] |

### 30% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.95 [0.9, 0.99] | 0.05 [0.01, 0.1] | 7.76 [6.74, 8.84] | 8.42 [7.28, 9.63] | 0.95 [0.9, 0.99] | 8.48 [7.33, 9.7] | 4.79 [4.39, 5.23] | 1.64 [1.28, 1.99] | 0.417 [0.37, 0.471] | 0.0212 [0.0154, 0.0277] | 577 [577, 577] |
| Isolation Forest alone | 0.82 [0.74, 0.89] | 0.04 [0.01, 0.08] | 9.68 [8.12, 11.4] | 12.8 [10.8, 15] | 0.82 [0.74, 0.89] | 13 [10.9, 15.3] | 7.61 [7.24, 8.01] | 1.19 [0.877, 1.52] | 0.477 [0.413, 0.551] | 0.00448 [0.00325, 0.00578] | 133 [133, 133] |

### 50% unreliable

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.92 [0.86, 0.97] | 0.02 [0, 0.05] | 9.15 [7.91, 10.4] | 10.1 [8.81, 11.4] | 0.92 [0.86, 0.97] | 10.1 [8.81, 11.4] | 4.35 [3.98, 4.76] | 1.45 [1.09, 1.8] | 0.434 [0.387, 0.492] | 0.0148 [0.0101, 0.0199] | 577 [577, 577] |
| Isolation Forest alone | 0.87 [0.8, 0.93] | 0.22 [0.14, 0.3] | 9.23 [7.65, 10.9] | 11.9 [9.95, 14.2] | 0.86 [0.78, 0.92] | 12.7 [10.7, 15] | 7.5 [7.2, 7.82] | 4.38 [3.89, 4.88] | 0.216 [0.191, 0.241] | 0.0188 [0.016, 0.0215] | 128 [128, 128] |

### All scenarios

| Method | Detection | Chance detection | Delay (detected) | Restricted delay | Attributable detection | Attributable delay | Alert size | False/yr | Precision | Null alert burden | Messages/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest + bacterial layer | 0.94 [0.913, 0.967] | 0.0433 [0.0233, 0.07] | 7.98 [7.34, 8.64] | 8.76 [8.06, 9.49] | 0.937 [0.907, 0.963] | 8.94 [8.2, 9.7] | 4.79 [4.54, 5.03] | 1.93 [1.71, 2.15] | 0.384 [0.36, 0.409] | 0.0231 [0.0197, 0.0266] | 577 [577, 577] |
| Isolation Forest alone | 0.803 [0.757, 0.847] | 0.0867 [0.0567, 0.12] | 9.67 [8.67, 10.7] | 13.6 [12.4, 14.9] | 0.8 [0.757, 0.843] | 13.9 [12.7, 15.3] | 7.59 [7.39, 7.8] | 1.88 [1.6, 2.19] | 0.37 [0.336, 0.411] | 0.00784 [0.00649, 0.00924] | 133 [132, 133] |