## Hasil utama (test set)

| model | type | mAP50 | mAP50-95 | mAP_small | mAP_medium | mAP_large | FPS |
|---|---|---|---|---|---|---|---|
| f_r50 | frcnn | 0.326 | 0.234 | 0.072 | 0.170 | 0.354 | 20.710 |
| y_s_default | yolo | 0.262 | 0.202 | 0.069 | 0.174 | 0.286 | 37.740 |

## AP@0.5:0.95 per kelas

| index | f_r50 | y_s_default |
|---|---|---|
| Plastic bag & wrapper | 0.265 | 0.229 |
| Cigarette | 0.062 | 0.037 |
| Bottle | 0.453 | 0.355 |
| Bottle cap | 0.318 | 0.317 |
| Can | 0.409 | 0.362 |
| Other plastic | 0.040 | 0.085 |
| Carton | 0.327 | 0.227 |
| Cup | 0.357 | 0.237 |
| Straw | 0.031 | 0.099 |
| Other | 0.076 | 0.069 |
