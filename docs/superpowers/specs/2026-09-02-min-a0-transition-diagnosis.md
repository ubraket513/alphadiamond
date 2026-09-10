# Min A0 전환 진단과 다음 실험

## 현재 브랜치 상태

GitHub 원격 기준으로는 **작업 브랜치가 남아 있지 않고, 보호된 `main` 하나만 존재**합니다. 열린 PR도 없습니다. 현재 `main` HEAD는 `b14c002b342880454819b398c7ad215087b524bb`이며, 마지막 변경은 A0 전환 게이트 결과를 기록한 문서 커밋입니다. 로컬 워크트리의 미커밋 변경 여부까지는 GitHub에서는 확인할 수 없습니다.

다만 `main`은 현재 완전히 green은 아닙니다. CI run #202에서 LibTorch CPU training, Linux/macOS/Windows core, Qt, sanitizers 등 실제 빌드·테스트 작업은 모두 통과했고, **`native-format`만 실패**했습니다. 여러 C++ 파일이 clang-format-18 기준을 만족하지 않는 것이 원인입니다. 즉 기능적 실패라기보다 정리되지 않은 통합 상태입니다.

훈련 상태는 다음과 같습니다.

- authoritative champion: iteration 100
- training step: 133,376
- active vacancy prior weight: `0.50`
- replay: 1,000,000 samples
- iteration 101: `SELF_PLAY` 중 의도적으로 중단
- iteration 101에서는 replay ingest와 learner update가 일어나지 않음
- 따라서 안전한 복구 지점은 iteration 100

최근 iteration-100 checkpoint를 prior weight `0.25`로 평가한 결과는 256판 중 12판 완주, 즉 **4.69% completion**이었습니다. 나머지 244판은 500수 cap에 걸렸습니다. A0 전환 기준인 97%와는 상당한 차이가 있습니다.

한 가지 운영상 중요한 불일치도 있습니다. 현재 committed `min-production-6h.json`은 여전히 `policy_loss_domain: "full"`이고 `bootstrap_prior_weight`가 명시되어 있지 않습니다. 반면 현재 durable run은 transition ledger를 통해 **legal loss + weight 0.50**으로 동작했습니다. 기존 run의 resume는 ledger가 보호하지만, repository config만 보고 새 run을 시작하면 현재 실험 조건을 재현하지 못합니다.

## 설계상의 결함인가?

제 판단은 다음과 같습니다.

> **현재 A0 실패는 MCTS 구현 결함보다는 B0→A0 학습 프로토콜과 데이터 계약의 설계 결함으로 보는 것이 맞습니다.**  
> 다만 아직 모델 아키텍처 자체의 결함이라고 단정할 증거는 없습니다.

정확하게는 코드가 계약대로 동작하지 않는 버그라기보다, **계약의 전제가 성립하지 않는 상태**입니다.

현재 설계는 대략 이런 순환을 기대합니다.

```text
vacancy prior가 게임을 완주시킴
→ MCTS visit target에 좋은 방향성이 담김
→ network policy가 그것을 학습
→ vacancy prior를 줄여도 network가 대신 방향을 제공
→ A0 전환
```

실제로 관측된 순환은 다릅니다.

```text
vacancy prior가 게임을 완주시킴
→ MCTS target은 여전히 매우 평평함
→ network는 미세한 방향성만 배움
→ prior를 줄이면 대부분 500수 동안 배회
→ 중단된 게임은 replay에서 제외
→ 실패 상태를 learner가 보지 못함
→ 다시 heuristic이 지원하는 성공 게임만 학습
```

이건 전형적인 **censored-data feedback loop**, 즉 실패 데이터가 구조적으로 검열되는 폐루프입니다.

### 근거 1: simulations 부족이 아닙니다

A0에서 serial MCTS를 128, 256, 400 simulations로 비교했지만 모두 **0/256 완주**였습니다. 256과 400은 행동을 고친 것이 아니라 계산량이 늘어 각 게임이 move cap 전에 180초 deadline에 걸리게 만들었을 뿐입니다. Adaptive 256/400도 0/256이었고, repetition trigger는 204,800수 중 사실상 두 번만 발동했습니다.

따라서 지금 parallel MCTS를 구현하면:

> 실패하는 탐색을 더 빠르게 또는 더 많이 실행하게 될 가능성

이 큽니다. 더 깊은 serial search가 품질을 개선한다는 전제가 이미 실패했으므로, parallel MCTS는 현재 문제의 해결책이 아닙니다.

### 근거 2: MCTS teacher target 자체가 지나치게 평평합니다

iteration-100의 alpha 0.25 probe에서:

- 평균 legal actions: 53.44
- effective actions: 47.25
- normalized target entropy: **0.971**
- 최대 action probability 평균: **0.057**
- top-3 mass: **0.136**

이었습니다. 즉 53개 legal action 중 약 47개가 실질적인 확률을 받고 있습니다. MCTS가 “몇 개의 좋은 수를 뚜렷하게 추천하는 teacher”라기보다 거의 균등 분포를 내놓고 있습니다.

이런 target에서는 vacancy prior가 만들어내는 작지만 누적적으로 중요한 방향성이 cross-entropy gradient에 강하게 남지 않습니다. 매 수마다 2~3% 정도의 미세한 방향 차이는 실제 게임 전체에서는 완주와 배회를 가르지만, learner 입장에서는 거의 균일한 분포의 작은 차이에 불과합니다.

### 근거 3: 실패 데이터가 정확히 버려집니다

현재 pipeline은 완료된 episode에 대해서만 `sample_from_move()`를 호출합니다. 중단된 episode는 마지막 진단 상태만 남기고, 그 안에서 수행된 수와 MCTS target은 training samples로 변환하지 않습니다.

최근 alpha 0.25 probe에서는 256판 중 244판이 중단됐으므로, learner가 실제로 고쳐야 할 상태 분포의 약 95%가 training contract 밖에 있습니다.

다만 **중단된 visit target을 그대로 학습시키는 것**도 답은 아닙니다. 그 target은 배회하다 실패한 policy가 만든 것이므로 그대로 모방하면 wandering을 강화할 수 있습니다. 실패 상태에는 별도의 더 나은 target이 필요합니다.

### 근거 4: legal loss 구현의 숨은 오류는 아닙니다

이 부분은 코드에서 따로 확인했습니다.

현재 `sample_from_move()`는 visit count가 0인 action도 포함해 **root의 모든 authoritative legal action**을 `sparse_policy`에 저장합니다. Trainer는 그 전체 집합을 legal mask로 사용해 legal-set log-softmax를 계산합니다. 즉 “방문된 action만 legal로 잘못 취급하는 문제”는 현재 코드에는 없습니다.

따라서 full-action loss 문제는 실제 결함이었고 legal loss로 고쳤지만, 그것만으로 A0 independence가 생기지는 않았습니다.

그리고 보고서의 `legal probability mass = 0.124`는 이제 원인 지표로 해석하면 안 됩니다. 실제 inference와 legal loss 모두 legal action 위에서 다시 정규화하므로, full 5,329-way softmax에서 legal action이 차지하는 절대 mass는 행동 선택에 직접 영향을 주지 않습니다. 지금 중요한 지표는:

- legal KL
- target entropy
- action별 progress signal
- no-prior completion

입니다.

## 아키텍처 결함 여부는 아직 미정입니다

현재 policy head는 node별 source/destination representation의 내적으로 action logit을 만듭니다. 이 구조와 6-block directional trunk가 vacancy heuristic의 방향성을 표현할 수 있는지는 아직 직접 측정하지 않았습니다.

따라서 다음 두 주장을 구분해야 합니다.

1. **표현은 가능하지만 현재 학습 target이 그 표현을 가르치지 못한다.**
2. **현재 architecture로는 해당 정책 자체를 충분히 표현하기 어렵다.**

지금 데이터는 1번을 강하게 의심하게 하지만, 2번을 배제하지는 못합니다.

이 구분을 위한 가장 정보량 높은 다음 실험은 **vacancy-prior realizability test**입니다.

## 다음 실험: vacancy prior를 직접 학습할 수 있는지 확인

iteration-100 checkpoint와 replay의 복사본을 사용해 production state를 전혀 변경하지 않는 offline diagnostic을 만드세요.

각 replay sample에는 이제 전체 legal action 집합이 들어 있으므로, feature channel 0에서 canonical self occupancy를 복원하고 같은 legal actions에 대해 `vacancy_prior()`를 다시 계산할 수 있습니다.

훈련 objective는 단순합니다.

\[
L_{\text{distill}}
=
KL(P_{\text{vacancy}}\parallel P_{\text{network,legal}})
\]

두 arm을 분리해야 합니다.

### Arm A: policy head만 학습

다음은 freeze합니다.

```text
input projection
residual trunk
output norm
value head
```

다음만 학습합니다.

```text
policy_source
policy_destination
```

### Arm B: trunk와 policy 전체 학습

value head만 freeze하고 trunk와 policy head를 함께 학습합니다.

측정할 것은 다음입니다.

- held-out vacancy KL
- network가 선택한 action의 기대 vacancy-potential 감소량
- teacher 대비 expected progress ratio
- top-1보다는 top-k teacher mass
- 학습 전후 policy KL
- network-prior-only 256-game completion
- p50/p90/p99
- seat balance
- target camp progress
- 실패 시 repetition이 아니라 stagnation인지

판정은 이렇게 하면 됩니다.

| 결과 | 해석 |
|---|---|
| head-only가 잘 맞고 A0가 개선됨 | architecture는 충분함. 현재 MCTS target/curriculum 설계가 문제 |
| head-only는 실패하지만 full network는 성공 | trunk representation이 현재 데이터에 맞지 않았지만 architecture 자체는 가능 |
| full network도 teacher를 못 맞춤 | policy head 또는 trunk 구조의 표현력 결함 가능성이 큼 |
| teacher fit은 좋은데 A0는 여전히 실패 | covariate shift와 error accumulation이 주원인 |

생산 전환 기준은 여전히 97%로 유지해야 하지만, 이 diagnostic의 1차 목적은 0% 수준에서 **의미 있는 completion 증가가 생기는지** 확인하는 것입니다.

## teacher fit이 성공하면

그때는 heuristic을 search에만 숨겨 넣지 말고, bootstrap 동안 learner에게 명시적인 auxiliary target으로 주는 것이 맞습니다.

```text
L = L_MCTS_visit
  + beta * KL(P_vacancy || P_network,legal)
  + L_value
```

`beta`를 별도로 anneal합니다.

이 방식은 최종 inference에 heuristic을 넣는 것이 아닙니다. 훈련 중에 heuristic을 network로 **직접 증류**한 뒤 제거하는 것입니다. 현재 방식은 heuristic → MCTS → visit count → network라는 간접 경로인데, target entropy가 0.97이면 중간 MCTS에서 teacher signal이 거의 씻겨 나갑니다.

## teacher fit은 되지만 A0가 계속 실패하면

그때는 recovery curriculum으로 넘어가야 합니다.

권장 방식은 실패한 visit distribution을 그대로 저장하는 것이 아니라:

1. alpha 0.25 또는 A0로 게임을 진행한다.
2. progress stagnation이나 move budget 근처에서 마지막 32~64개 상태를 보존한다.
3. 그 상태부터 alpha 0.50 또는 1.00의 rescue search로 실제 terminal까지 진행한다.
4. rescue가 완주한 경우에만 해당 상태의 policy target과 terminal placement value를 채택한다.
5. 전체 minibatch 중 recovery sample 비율을 제한한다.

이 방식은 learner가 실제 실패 상태를 보게 하면서도, 실패 policy 자체를 모방하지 않습니다. 일종의 DAgger-style expert fallback입니다.

Raw aborted trajectory를 그대로 policy-only sample로 넣는 방식은 첫 선택으로 권하지 않습니다. 현재 A0 target이 거의 uniform이므로 wandering을 고착할 위험이 있습니다.

## value 쪽도 한 번은 확인해야 합니다

128→400 simulations가 아무 효과가 없었다는 건 search가 더 많은 leaf를 보는 것만으로는 판단이 좋아지지 않는다는 뜻입니다. 다음 진단에는 root에서 다음을 추가하는 게 좋습니다.

```text
Q range / standard deviation
PUCT exploration U range
Q와 U의 비율
최종 승자 rank prediction accuracy
게임 진행 구간별 value MSE
```

초기 value loss가 약 0.58~0.61이었는데, 항상 `[0,0,0]`을 예측하는 기준 MSE가 약 0.667입니다. 즉 value는 완전히 죽지는 않았지만 장기 3P 결과를 강하게 구분한다고 보기도 어렵습니다. Q가 U에 비해 거의 평평하면 MCTS는 사실상 prior만 따라가므로, simulations를 늘려도 달라지지 않는 것이 자연스럽습니다.

## 지금 바로 정리할 세 가지

첫째, `native-format`만 수정한 작은 PR로 main을 다시 green으로 만들어야 합니다. 현재 실패는 코드 의미와 무관하지만, 다음 실험의 clean baseline을 위해 필요합니다.

둘째, 현재 effective configuration을 별도 versioned config로 커밋하는 편이 좋습니다.

```text
configs/alphazero/min-anneal-alpha050-v1.json
```

여기에는 최소한 다음이 명시되어야 합니다.

```json
{
  "self_play": {
    "bootstrap_prior": "canonical-target-vacancy-distance-v2",
    "bootstrap_prior_weight": 0.5
  },
  "training": {
    "policy_loss_domain": "legal"
  }
}
```

기존 durable run의 immutable config를 덮어쓰는 것이 아니라, repository source of truth에서도 현재 운영점을 재현 가능하게 만드는 파일입니다.

셋째, 최신 alpha 0.25 raw probe는 build provenance가 `source_commit=2b4ae24...`, `dirty=true`로 기록돼 있습니다. 결과를 뒤집을 문제는 아니지만 다음 비교에서는 허용하면 안 됩니다. clean HEAD에서 재configure/rebuild하고 다음을 gate로 두는 게 좋습니다.

```text
reported source_commit == git rev-parse HEAD
reported dirty == false
active config digest == expected digest
checkpoint model digest == expected digest
```

## 최종 권고

현재 우선순위는 이렇게 두는 게 맞습니다.

1. formatting과 effective-config provenance 정리
2. **vacancy-prior realizability / direct-distillation diagnostic**
3. 결과에 따라 auxiliary distillation 또는 recovery curriculum
4. 그 뒤에 value/Q-vs-U 진단
5. full-network teacher fit이 실패할 때만 architecture 변경
6. parallel MCTS는 보류

따라서 현재 증상은 **“Min 모델 구조가 틀렸다”가 아니라 “heuristic-supported 성공 데이터만으로 heuristic-free policy가 자연스럽게 생길 것이라는 transition design이 틀렸다”**고 보는 것이 가장 정확합니다. 지금은 계산량을 늘릴 단계가 아니라, teacher signal을 learner에게 직접 전달하고 실패 분포를 학습 계약 안으로 넣을 단계입니다.
