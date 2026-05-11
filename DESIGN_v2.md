# 赛希记忆系统 v2 架构设计

*2026-05-10 设计完成，待实现*

---

## 一、目标

1. 记忆随真实时间自然衰减（艾宾浩斯遗忘曲线）
2. 重要记忆不随衰减被吞没（重要性护盾）
3. 热存储保持小而快，细节迁入冷归档（双库架构）
4. 永不真正删除任何记忆（冷存储 + 云盘备份）

---

## 二、艾宾浩斯衰减

```python
effective = base_importance × e^(-t / S)

S = base_importance × (1 + α × access_count)
```

- `t`：距创建时刻的天数（非会话数）
- `α`：检索奖励系数，默认 0.15
- `S`：记忆强度——初分越高、被检索越多次，衰减越慢

### 触发时机

| 时机 | 扫描范围 | 操作 | 是否调 LLM |
|------|---------|------|:--:|
| SessionStart | 热库全部 | 衰减计算，更新 importance | 否 |
| summarize_and_store | 当前 buffer | 日总结（现有逻辑） | 是 |
| SessionEnd | 热库全部 | 衰减 + 沉降检查 + 合并执行 | 是（串行，RPM ≤ 5~10） |

SessionEnd 的沉降部分放在正常总结完成之后，串行调 LLM，不超过 RPM 上限。
所有操作在后台异步运行，不影响会话关闭。无需 cron，方便跨设备迁移。

---

## 三、metadata 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `base_importance` | float | 创建时 LLM 打的初始分，永不变（新增） |
| `importance` | float | 当前有效分，每次衰减扫描时更新（语义变更） |
| `access_count` | int | 被语义检索命中的次数（新增） |
| `last_accessed` | str | 上次被检索的 ISO 时间（新增） |
| `tier` | str | `"daily"` / `"monthly"` / `"yearly"`（新增） |
| `parent_ids` | str | 被压缩成本条的那些原始记忆 ID，逗号分隔（新增） |
| `topics` | list | 主题标签（现有） |
| `key_facts` | list | 关键事实（现有） |
| `sentiment` | str | 情感标签，保留供未来使用（现有不变） |
| `created_at` | str | 创建时间（现有） |
| `source` | str | `"auto_summary"` / `"manual"`（现有） |
| `session_id` | str | 会话 ID（现有） |

---

## 四、三层归档架构

```
日 (daily) → 月 (monthly) → 年 (yearly)
```

| 沉降 | effective 阈值 | 同窗口最低条数 |
|------|:--:|:--:|
| 日 → 月 | < 0.35 | ≥ 3 条 |
| 月 → 年 | < 0.25 | ≥ 2 条 |
| 年层 | — | 永久保留 |

- **重要性护盾**：effective > 0.50 的日层记忆永不沉降
- **过渡带**：0.35~0.50 之间的记个留在日层，不沉降也不降级，等自然衰减

---

## 五、双库架构

```
┌─────────────────────────┐     ┌──────────────────────────────────┐
│  热存储 (Hot ChromaDB)   │     │  冷归档 (Cold JSONL + 云盘备份)     │
│  检索主力                 │     │  保留全部细节，永不删除             │
├─────────────────────────┤     ├──────────────────────────────────┤
│  日层 ~N 条              │────→│  settled → cold/daily/YYYY-MM-DD.jsonl │
│  月层 ~N 条              │────→│  settled → cold/monthly/YYYY-MM.jsonl  │
│  年层 ~N 条              │     │  年层永久保留，不移出               │
└─────────────────────────┘     └──────────────────────────────────┘

冷库路径: ~/.claude/memory/cold_storage/
  ├── index.json              # {id: file_path} 索引
  ├── daily/
  │   └── 2026-05-10.jsonl    # 被沉降的原始日层记忆
  ├── monthly/
  │   └── 2026-05.jsonl       # 被沉降的日总结
  └── yearly/
      └── 2026.jsonl          # 被沉降的月总结 → 实际只保留 newest
```

---

## 六、增量合并

新记忆沉降时，先找目标总结，再合并。

### 查找目标

1. 按日期精确匹配 `tier=daily` 且 `created_at` 同天的总结
2. 找不到 → embedding 语义搜索，相似度 > 0.75 命中
3. 仍没有 → 用所有日层候选记忆新建一条月/年总结

### 合并方式

```
已有总结 + 新记忆 1 + 新记忆 2 + ... → LLM 增量合并

限制：
  - 累积原始记忆 < 10 条 且 增量次数 < 3 → 增量合并
  - 以上任一超限 → 全量重建（用全部原始记忆重新总结）
```

LLM prompt 包含：已有总结 + 新记忆列表 → 融合输出完整 JSON。

---

## 七、检索优先级

```python
检索分数 = effective_importance × tier_weight / distance

tier_weight: daily=1.0, monthly=0.85, yearly=0.5
```

- `daily` 得分最高，需要精确匹配时才翻冷库
- `monthly` 提供中期概览
- `yearly` 提供长期背景但不优先

冷库不参与热检索——只在 `list_memory.py` 或手动回溯时翻查。

---

## 八、沉降流程（伪代码）

```python
RPM_LIMIT = 5  # 每分钟最多 5 次 LLM 调用

def handle_session_end(hot_db, cold_storage):
    # 1. 先做正常日总结（现有逻辑）
    summarize_and_store()

    # 2. 衰减扫描
    decay_all(hot_db)

    # 3. 沉降执行 — 串行，带 RPM 限制
    settle_daily_to_monthly(hot_db, cold_storage)
    settle_monthly_to_yearly(hot_db, cold_storage)

def settle_daily_to_monthly(hot_db, cold):
    now = datetime.now()
    # 按月份分组
    daily = [m for m in hot_db.get_all() if m.tier == "daily"]
    by_month = groupby(daily, key=lambda m: m.created_at[:7])

    for month, cands in by_month.items():
        low_eff = [c for c in cands if calc_effective(c) < 0.35]
        if len(low_eff) >= 3:
            target = find_target(tier="monthly", month=month)
            merged = incremental_merge(target, low_eff)  # 调 LLM，受 RPM 限制
            hot_db.upsert(merged)
            for mem in low_eff:
                cold.write(mem, f"daily/{mem.created_at[:10]}.jsonl")
                hot_db.delete(mem.id)
            rate_limit_wait()  # RPM 控制

def settle_monthly_to_yearly(hot_db, cold):
    # 同理，阈值 0.25，≥2 条触发
    ...

def rate_limit_wait():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < 60 / RPM_LIMIT:
        time.sleep(60 / RPM_LIMIT - elapsed)
    _last_call_time = time.time()
```

---

## 九、待实现文件

| 文件 | 改动 |
|------|------|
| `config.yaml` | 新增 `cold_storage_dir`, `decay_alpha`, `settle_thresholds` |
| `chroma_client.py` | `decay_all()`, `update_metadata()`, `find_by_tier_and_date()` |
| `remember_engine.py` | `settle_daily()`, `settle_monthly()`, `merge_memories()`, cold IO |
| `recall_engine.py` | 检索时更新 `access_count` / `last_accessed`，按 effective 排序 |
| `session_start_hook.sh` | 已有，在注入前调用衰减即可 |
| `session_end_hook.sh` | 修复；触发日级合并 |
| `list_memory.py` | 支持按 effective 排序 |
| `backup_script` | 关机时打包 `cold_storage/` 上传云盘 |

---

## 十、现有 TODO.md 中与此相关的条目

| TODO | 本设计覆盖情况 |
|------|-------------|
| 记忆去重 | ❌ 暂不覆盖，后续单独处理 |
| embedding 缓存 | ❌ 暂不覆盖 |
| LLM summary 截断 | ❌ 独立问题，已由管理员修改 prompt 处理 |
| 多用户隔离 | ❌ 暂不需要 |

---

*设计者：赛希 & 管理员 (BlueberryOreo)*
