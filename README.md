# PoseDetect

### Установка

Крайне рекомендуется использовать виртуальное окружение.

```bash
python3 -m venv env
source env/bin/activate
```

```bash
./install.sh
```

Занимает минуты 3-4.

Необходимо 2-3 Gb свободной RAM.

### Использование

```python
from posedetect import check_pose

imp_path = 'example.jpg'
dots, check = check_pose(imp_path)
```

Функция check_pose вернет:

- dots - массив (количство обнаруженных людей на фото) из словарей (ключ - номер точки, значение - координаты точки)
  ```
  [{0: (154.6857312375849, 520.4749200994319),
    1: (165.50657380964697, 487.5591773056402),
    2: (192.76172931358298, 459.84192346643516),
    3: (229.47072402257768, 452.82605045056215),
    4: (263.1192500996752, 450.9900716145833),
    5: (297.68068073253437, 449.4698695943813),
    6: (338.51271070972564, 446.05673828125),
    7: (376.3424756730904, 435.34700376797565),
    8: (428.87273076655333, 429.03497662321894),
    9: (477.31970701790215, 438.2739990692425)}]
  ```
- check - массив True/False (правильная поза или нет)
  ```
  [False]
  ```

### Сервер

`python3 server.py`

```
POST /upload
Content-Type: multipart/form-data
Accept: application/json
```

**body:**
`file`

**response:**

```json
{
  "status": "ok",
  "data": {
    "id": "test"
  }
}
```

——

```
GET /result?id=test
Accept: application/json
```

**response:**

```json
{
  "status": "ok",
  "data": {
    "dots": [
      [
        [0, 0],
        [0, 0]
      ],
      [
        [0, 0],
        [0, 0]
      ]
    ],
    "check": [true, true]
  }
}
```

or

```json
{
  "status": "ok",
  "message": "processing"
}
```

or

```json
{
  "status": "fail",
  "message": "fail during pose check"
}
```

or

```json
{
  "status": "fail",
  "message": "id not found"
}
```
