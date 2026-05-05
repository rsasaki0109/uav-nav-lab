# uav-nav-lab — Plan & Roadmap

> **位置付け**: `docs/findings.md` は *終わった研究の記録*、`README.md` は
> *入口とハイライト*、この `plan.md` は *これから何をやるか / なぜやるか*
> をまとめる作戦ノート。短期 (次の 1-3 PR)・中期・長期に分けて書く。
> 思いつきや却下案も「やらない理由」と一緒に残しておく — 後から
> 「なぜこの方向に行かなかったか」を辿れるようにするため。
>
> 最終更新: 2026-05-05 (PR #48 準備中: short-term B/D/A/C すべて実装済み)

---

## 1. これまでの到達点 (2026-05-03 時点)

83 commit / 47 PR まで来た。骨子は以下の通り完成している。

### 1.1 フレームワーク本体

- **YAML 駆動の実験ランナー** (`uav-nav run` / `uav-nav sweep` /
  `uav-nav eval` / `uav-nav viz` / `uav-nav video`)。Cartesian product
  sweep + Wilson 95 % CI が標準装備。
- **5 軸プラガブル**: `sim` / `scenario` / `planner` / `sensor` /
  `predictor` を `@REGISTRY.register("name")` + `from_config(cfg)`
  だけで足せる。
- **マルチドローン runner** (`runner/multi.py`) — 1 シナリオ N ドローン、
  各機独立の sim/sensor/planner、ピア obs 経由で相互回避。
  PR #47 で AirSim の共有クロックにも対応 (master-handoff)。

### 1.2 バックエンド一覧

| 軸          | 実装                                                                    |
|-------------|-------------------------------------------------------------------------|
| `sim`       | `dummy_2d` / `dummy_3d` (point-mass) / `airsim` / `ros2`                |
| `scenario`  | `grid_world` / `voxel_world` / `multi_drone_grid` / `multi_drone_voxel` |
| `planner`   | `straight` / `astar` / `rrt` / `rrt_star` / `chomp` / `mpc` / `mpc_chomp` / `mppi` |
| `sensor`    | `perfect` / `delayed` / `kalman_delayed` / `lidar` / `pointcloud_occupancy` / `depth_image_occupancy` |
| `predictor` | constant velocity (planner 内蔵 `use_prediction` で ON/OFF)              |

### 1.3 主要研究結果 (`docs/findings.md` に詳細)

| トピック                    | 結論                                            | PR     |
|-----------------------------|-------------------------------------------------|--------|
| MPC compute Pareto (2D)     | `n_samples=16, h=20` 単独 Pareto                 | #11    |
| 3D Pareto                   | `n_samples` 選好が 2D と逆転                    | #15    |
| 知覚遅延クリフ (2D/3D)      | 同じコーナーで崩れる、3D は escape volume で軟化 | #19/#46|
| 多機 N スケール             | N 増で peer prediction Δ が拡大                 | #28    |
| 3D escape volume            | 体積でかいと Δ が消える                         | #39    |
| 3D 密度                     | 密度上げると Δ 復活                             | #41    |
| 3D peer-prediction 削除     | 「予測無し」は密度 8x より悪い                   | #43    |
| 風 miscalibration           | planner の風モデル = sim 真値が最強              | #29    |
| MPC + CHOMP 重ね掛け        | 飽和済み MPC 上では wash (誠実な null)           | #21    |
| action-jump cost            | 既存 `w_smooth` チューニングが層追加に勝つ      | #38    |
| RRT* (asymptotic optimal)   | フレッシュネス勝負ではプレーン RRT に負ける     | #40    |
| AirSim 統合                 | E2E 動作 (LiDAR / depth / camera / multi)        | #44/#45/#46/#47 |

### 1.4 残タスク (細かいやつ)

- `tests/test_smoke.py::test_multi_drone_voxel_anim_groups_drones_per_episode` が
  matplotlib `Axes3D` の import 衝突で local fail (CI は通っている)。
  影響軽微なので放置中。
- `roadmap` 節 (README) の「3D perception-latency 再検証」は #46 で
  片付いた → 次の README 触るときに削る。
- `roadmap` 節の「ROS 2 sim-time 対応」は #30 で済んでいる → 同様に削る。
  → **次の PR ついでに README roadmap を整理する**。

---

## 2. 短期 (次の 1-3 PR で片付けたい)

### 2.1 候補 A: AirSim multi-drone を「一段詰める」 — passive-first 順序

**動機**: PR #47 では「同一高度 4-way 中央交差」が AirSim の 1-tick
command lag のせいで脆く、altitude を ±2 m staggered にして逃げた。
本質的には runner のループ順序を `[1, 2, 3, ..., 0]` (passive 先 →
master 最後) に並び替えるだけで lag は消えるはず。

**やること**:
1. `runner/multi.py` の step 2 ループ順序を「passive bridge を先に
   `moveByVelocityAsync` させてから、master が `simContinueForTime`
   する」順に並び替える。
2. ただし passive sim の `getMultirotorState` (ステート読み戻し) は
   master の continue の *後* で読む必要があるので、step を
   「コマンド発行フェーズ」と「ステート読み戻しフェーズ」の 2 段に
   分割する必要があるかもしれない。
3. `examples/exp_airsim_multi_demo.yaml` の altitude を全部 30 m に
   戻して、同一高度 4-way 中央交差が成功するか検証。
4. dummy_3d の multi runner には影響しないことを smoke test で確認。

**やらない理由 / リスク**:
- ループ順序の入れ替えだけでなく step 分割も要るかもしれず、思った
  以上に runner の見通しが悪くなる可能性。
- staggered altitude は「AirSim の物理を逃げてる」というより
  「demo の見せ方として 4 機が画面内で違うレイヤーに居た方が見やすい」
  という副次効果もあった。
- **判定**: lag 解消は技術的に正しいが demo 体験は今で十分。
  優先度: 中。

### 2.2 候補 B: AirSim で sensor-latency cliff を再現 (PR #43/#46 の AirSim 版)

**動機**: PR #46 で「同じ MPC plan を dummy_3d と AirSim で走らせて
物理差を可視化」する transferability ablation が完成した。次の自然な
段階は「dummy_3d で発見した *研究結果* (cliff / wind miscal / peer
prediction Δ) が AirSim 上でも同じ形で出るか」を検証すること。
シミュレータ間転移性の研究の第二弾。

**ターゲット**:
1. **Perception-latency cliff** (PR #19/#46): `delayed` センサーの
   `latency_steps` を 0..6 振る ablation を AirSim で走らせて、
   dummy_3d で見えた「3-4 step で崖、ego_extrapolation で +50 pp 戻る」
   が AirSim でも見えるか。
2. もし見えれば → 「研究結果が転移する」ポジティブ result。
3. もし見えなければ → AirSim 物理の何が崖を埋めている / 動かしている
   かを切り分け (motor lag? air drag? pitch coupling?)。これも
   transferability 研究としては面白い。

**コスト見積**:
- AirSim を起動して 7 セル × n=10 episode = 70 episode。
  1 episode ~10 s として ~12 min/全条件。許容範囲。
- 既存の `delayed` sensor + `voxel_world` がそのまま AirSim で
  動くはずなので、新規コードはほぼ YAML だけ。

**判定**: 優先度 **高**。AirSim 統合の研究的 value を最大化する筋。

### 2.3 候補 C: AirSim で wind miscalibration ablation を再現

**動機**: PR #29 の「planner の風モデル = sim 真値が最強、awareness
だけでは physics に勝てない」という結論を AirSim の `Wind` 設定で
再現できるか。

**やること**:
1. AirSim の `settings.json` に `Wind: {"X": Vx, "Y": Vy, "Z": Vz}` を
   入れて風を吹かせる。
2. planner 側の `wind_belief` を sim 真値とずらした 3x3 grid
   (under/exact/over) で sweep。
3. dummy_3d で見えた「belief = reality でしか success が立たない」が
   AirSim でも成立するか。

**コスト**: 中。AirSim の Wind は静的設定 (動的に変えるには Plugin
書く必要あり) なので、9 セル × n=10 episode = 90 episode、設定毎に
AirSim 再起動が要る → ~1 時間ぐらい。

**判定**: 優先度 中。B と同じ「研究の AirSim 転移」筋だが B より重い。

### 2.4 候補 D: README roadmap の整理 + findings.md に PR #47 の節追加

**動機**: 既に解決済みの roadmap 項目 (3D perception latency 再検証 /
ROS 2 sim-time) が README に残っている。地味だが残すと混乱の元。
ついでに PR #47 を `docs/findings.md` の章として書き起こしておく
(AirSim 共有クロックの罠と master-handoff 話)。

**やること**:
1. README の `## 🗺️ Roadmap` 節を更新 — 既済 2 件削除、新項目を追加
   (AirSim transferability 第二弾, とか)。
2. `docs/findings.md` に "AirSim multi-drone: shared physics clock and
   master handoff" 節を追加。実験というより *エンジニアリング知見*
   の記録。

**判定**: 優先度 低だが、B/C を始める前にここを片付けるのが綺麗。
B/C どちらかと一緒のコミットでもよい。

---

## 3. 中期 (次の 5-10 PR)

### 3.1 「研究結果の AirSim 転移」シリーズ (上記 B/C の発展)

PR #43〜#46 で「フレームワークが AirSim 上で動く」「同じ plan を両方で
走らせる」までは終わっている。次の章は **「dummy_3d で得た定性的結論が
AirSim でも保たれるか」を一個ずつ潰していく** こと。候補:

- [ ] perception-latency cliff (B)
- [ ] wind miscalibration (C)
- [ ] peer-prediction の Δ (4 機 N=4 で AirSim multi-drone)
- [ ] sensor FOV ablation (omni-LiDAR vs forward depth) — AirSim 版
- [ ] CHOMP+RRT-init の +17 pp が AirSim でも出るか (sim-transfer 第二弾)

これらが全部「転移する」なら **dummy_3d は AirSim の良い proxy**
という強いメッセージが出せる。逆に「転移しない」項目があるなら、
それ自体が「dummy_3d のどこを直せば AirSim と等価になるか」という
別の研究テーマになる。

### 3.2 ROS 2 ↔ AirSim 統合

現状 `ros2_bridge` は完成済み (PR #30)。次のステップは:

- AirSim の `ros2/AirsimROSWrapper` を立てて、
  `airsim_bridge` (直結) と `ros2_bridge` (ROS2 越し) を同じ
  `voxel_world` で走らせ、E2E 数値が一致することを確認。
- これができると「ROS2 越しでも数値が変わらない = bridge 越しの
  オーバーヘッドが研究結果を歪めない」ことが言える。

**コスト**: 大。AirSim ROS2 wrapper のセットアップが必要。

### 3.3 GPU MPC / MPPI

現状 MPC/MPPI は CPU n=16 サンプル前提。GPU を使うと n=128/256 が
sub-ms に乗るはずで、これによって「Pareto コーナー」が右にずれる。

候補ライブラリ:
- `pytorch` ベースの自前 MPPI
- `mppi-isaac` (NVIDIA)
- 既存の `examples/exp_predictive.yaml` の Pareto を GPU 版で再描画

**判定**: 興味はあるが、本フレームワークの売りは「シンプル + プラガブル」
なので、GPU 依存を持ち込むのは慎重に。`gpu_mppi` という別 planner
として分離するのが筋。

### 3.4 RL 比較ベースライン

「古典 plan-based」 vs 「学習ベース」を同じ scenario で走らせる比較が
無い。`stable-baselines3` で SAC/PPO エージェントを `voxel_world` で
学習させて、Pareto-MPC vs RL の Pareto curve を出す。

**判定**: 学習側のセットアップ (state/action space, reward shaping,
ロールアウト時間) でフレームワークが歪む可能性が高い。やるなら
別 repo か submodule で。優先度低。

---

## 4. 長期 (構想だけ)

### 4.1 実機転移 (sim-to-real)

ROS2 bridge は MAVROS 経由で PX4 SITL → 実機まで一応繋がる。
ただし「研究の」実機転移は scope を絞らないと sink になる:

- 屋内 OptiTrack で position feedback を入れた perception clean
  setup での Pareto-MPC 動作確認 → 中期
- 屋外 GPS で sensor-latency cliff の実機再現 → 長期 (実機要件が重い)

### 4.2 マルチエージェント学習

PR #43 で示した「CV peer prediction が密度より効く」を出発点に、
学習ベースの peer predictor (LSTM / Transformer) と CV を比較する。

### 4.3 公開化

論文 / preprint / OSS としての宣伝。現状の研究結果はそれだけで
*短いワークショップ論文* 1 本にできる量がある:
- "Compute-aware planner Pareto on a unified 2D/3D dynamic-obstacle
  benchmark"
- "Constant-velocity peer prediction is the dominant axis in
  multi-drone coordination — density and prediction model studies
  on `multi_drone_voxel`"

---

## 5. 次の打ち手 (2026-05-05 時点)

短期 B/D/A/C はすべて実装済み。結果:

1. ~~**候補 B (AirSim sensor-latency cliff)**~~ → **完了。cliff は転移せず。**
   AirSim の motor ramp が機械的ローパスフィルタとして働き、stale position
   による velocity コマンド振動を平滑化するため。dummy 側の検証で cliff 自体
   は実在確認済み。→ `docs/findings.md` に記録。
2. ~~**候補 D (README roadmap 整理 + findings.md 更新)**~~ → **完了。**
   README 既済 2 件削除、新規 3 件追加。findings.md に PR #47 章追加。
3. ~~**候補 A (AirSim multi-drone passive-first 順序)**~~ → **完了。**
   Bridge を `step_command` / `step_readback` に分割、runner で two-phase
   passive-first dispatch。1-tick lag は解消したが、uniform-altitude 4-way
   中央交差は AirSim の mesh collision で破綻。staggered altitude は引き続き
   必要。2-drone uniform では 100% success 確認。
4. ~~**候補 C (AirSim wind miscalibration)**~~ → **完了。転移せず。**
   SimpleFlight の velocity controller が 5 m/s 風を完全に打ち消す。
   MPC の wind belief と競合し、no-awareness が最速という逆転結果。
   AirSim wind 対応のために bridge に `simSetWind` 追加。

**次の優先順:**

1. **中期 1: ROS 2 ↔ AirSim 統合検証** — `airsim_bridge` (直結) と
   `ros2_bridge` (ROS2 越し) を同じ `voxel_world` で走らせ数値一致確認。
2. **中期 2: GPU MPC / MPPI** — n_samples=128/256 で Pareto 再描画。
3. **B の延長: 限界条件 cliff** — max_speed=15, 密度 >10% で AirSim での
   cliff 出現可否を検証（SimpleFlight の速度限界が課題）。

→ 次に着手するなら **中期 1 (ROS 2 ↔ AirSim)** が筋。

---

## 6. やらないと決めたもの (供養コーナー)

| 案                                       | 却下理由                                      |
|------------------------------------------|-----------------------------------------------|
| MPC を CasADi/IPOPT に置き換え            | フレームワークの「シンプル」を壊す。学術的価値も既存 sampling MPC で十分出ている。 |
| 全 planner に GPU 必須化                  | フレームワークのポータビリティを壊す。`gpu_mppi` を別 planner として足すなら可。 |
| `gym.Env` ラッパー                        | RL 比較を本気でやる時に検討。今は不要。      |
| Web UI / dashboard                        | scope 外。CLI + matplotlib + GIF で十分。     |
| `omegaconf` への移行                      | 現 `ExperimentConfig` で困ってない。         |
| collision 以外の per-step reward 設計     | フレームワークが planner 比較ツールとしての性格を失う。 |
