# Alpha Temporal Adaptation

Эта поддиректория содержит вспомогательные материалы для подключения датасета Alpha Temporal к пайплайну GenTKG.

## Структура данных

Ожидается, что рабочие файлы располагаются в `data/alpha_temporal/` и повторяют формат стандартных наборов:

```
data/alpha_temporal/
  train.txt
  valid.txt
  test.txt
  all_facts.txt
  entity2id.json
  relation2id.json
  ts2id.json
  meta.json          # опционально, но удобно задаёт period/num_relations
```

Пример содержимого `meta.json` смотрите в `meta.json.example`. Скопируйте файл и при необходимости скорректируйте значения:

```
cp adaptations/alpha_temporal/meta.json.example data/alpha_temporal/meta.json
```

## Конвертация из synthetic_nodes/synthetic_edges

Если исходные данные лежат в `data/synthetic_nodes.parquet` и `data/synthetic_edges.parquet`, используйте скрипт `build_from_synthetic.py`:

```
python adaptations/alpha_temporal/build_from_synthetic.py \
  --nodes data/synthetic_nodes.parquet \
  --edges data/synthetic_edges.parquet \
  --dataset alpha_temporal \
  --write_meta
```

Скрипт создаёт `data/alpha_temporal/{train,valid,test,all_facts}.txt` и словари id. Деление выполняется по временным отметкам: все срезы, кроме двух последних, попадают в train, предпоследний — валидация, последний — тест. Для быстрых экспериментов ограничьте число временных слоёв флагами `--train_splits`, `--valid_splits`, `--test_splits` или заранее отфильтруйте parquet.

> Предупреждение: на полном наборе (~2 млн рёбер) конвертация и последующий TLR-прогон занимают часы и требуют заметной памяти.

## Генерация истории и ответов

После подготовки данных выполните извлечение истории TLR:

```
python data_utils/retrieve.py \
  --dataset alpha_temporal \
  --name_of_rules_file <rules.json> \
  --splits train,valid,test \
  --data_root ./data \
  --output_root ./output \
  --processed_root ./data/processed_new
```

Если значения периода или количества отношений отличаются от `meta.json`, задайте их явно:

```
python data_utils/retrieve.py --dataset alpha_temporal --period 48 --num_relations 512
```

## Конвертация в формат обучения

Для подготовки JSON-файлов используйте стандартный скрипт:

```
python data_utils/create_json_train.py \
  --dir_of_trainset ./data/processed_new/alpha_temporal/train/history_facts/history_facts_alpha_temporal.txt \
  --dir_of_answers ./data/processed_new/alpha_temporal/train/test_answers/test_answers_alpha_temporal.txt \
  --dir_of_entities2id ./data/alpha_temporal/entity2id.json \
  --path_save ./data/processed_new/alpha_temporal/train \
  --name_train alpha_temporal
```

## Быстрая проверка

Скрипт `write_meta.py` из этой же директории помогает сгенерировать `meta.json` при известных параметрах:

```
python adaptations/alpha_temporal/write_meta.py --period 24 --num_relations 320 \
  --dataset_root ./data --dataset alpha_temporal
```

После этого можно запускать fine-tuning и инференс так же, как для других датасетов.
