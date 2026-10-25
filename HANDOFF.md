# 인수인계 (HANDOFF) — Blender 블루프린트 재구성 스킬

마지막 업데이트: 2026-10-25 · 이전 작업: 로컬 Claude Code 세션 `2710d383` (2026-10-22 ~ 2026-10-23 13:31 KST, 주간 사용량 한도에 걸려 중단)

## 1. 프로젝트 목표

- 처음 목표: [prompt.md](prompt.md)를 바탕으로 스킬을 만든다. 캐릭터 이미지를 넣으면 같은 모양의 Blender 모델이 나와야 한다.
- 지금 걸려 있는 `/goal`: "에이전트가 정면·측면·후면 이미지만으로 모델링을 잘 하도록 연구해서 완벽한 스킬을 만든다."
- 스킬 본체: [blender-blueprint-reconstruction/](blender-blueprint-reconstruction/)
  - 에이전트 지침: [SKILL.md](blender-blueprint-reconstruction/SKILL.md)
  - 사용법과 레이아웃: [README.md](blender-blueprint-reconstruction/README.md)
- 설계 원칙: 에이전트는 의미만 정한다(부품, 라벨, 관계). 보정, 측정, 삼각측량, 피팅, 렌더, 검증 같은 수치 작업은 도구가 한다.

## 2. 완료된 작업

| 영역 | 상태 |
|---|---|
| 파이프라인 | ViewSpec → SceneSpec → 결정론적 빌더 → 정사영 렌더 비교 → 보정 루프 → report.json |
| 하드서피스 트랙 | box, rounded_box, cylinder, lathe, plate, frame 등 + Mirror, Boolean, Bevel. 텍스처 투영 지원 |
| 캐릭터 트랙 | silhouette_hull 몸체, 라벨 기반 부품(cone, blade, disc, gem, capsule, sphere, strand, plate), 앵커 삼각측량, 부품 자동 피팅, layer_hull(얼굴 레이어), 대칭 가상 뷰 |
| 레퍼런스 준비 CLI | `prepare_references.py`: split-sheet, masks, draft-view-spec, check(일관성 게이트), measure, propose, labels, anchors |
| 검증 | 바닥값(floor) 대비 초과 오차, 부품 IoU, 오귀속, 가림, 홀드아웃 뷰, ring/float 플래그 |
| 외부 스킬 통합 | EL4CTEO/blender-skills(MIT)의 hull 후처리 이식 ([THIRD_PARTY_NOTICES.md](blender-blueprint-reconstruction/THIRD_PARTY_NOTICES.md)) |
| MCP API | `scripts/bbr/api.py`. 모든 함수가 `ok` dict를 반환하고 예외를 던지지 않음. 입력 검증, update_component 실패 시 롤백 |
| 벤치 | `bench/run_bench.py`, 17개 케이스. 마지막 실행 `final5`는 전부 통과하고 final4 대비 회귀 없음 → [docs/bench-final5-summary.md](docs/bench-final5-summary.md) |
| 테스트 | WI-1 착수 전 기준 pytest 459개 통과, Blender 스모크 `SMOKE OK` |
| SKILL.md | 약 2000단어에서 988단어로 줄임(한도 1000단어). 캐릭터 요약 문서 `references/character-cheatsheet.md` |

### 블라인드 테스트 추이

방식: SKILL.md만 받은 새 에이전트가 처음 보는 대상을 모델링한다. 실행 횟수와 시간에 제한이 있다. 프로토콜은 `bench/blind_run.md`에 있다.

| 회차 | 대상 | 결과 | 주된 문제 |
|---|---|---|---|
| 1 | SD 마법사 | 6.1% | 도구 결함 |
| 2 | 여우 소년 | 5.9% | 도구 결함 |
| 3 | 펭귄 셰프 | 2.3% | 기능 부족 |
| 4 | 보물상자(하드서피스) | 2.6% | 하드서피스 텍스처 |
| 5 | 버섯 집 | 약 4.2% | 기능 부족 |
| 6 | 트윈테일 | 11.6%로 실패 → 수정 후 7.8%로 통과 | 붙어 있는 부속, 머리 다발 |
| 7 | 아기 드래곤 | 실패 → 수정 후 통과 | 막 형태 날개(`plate`), 떨어진 파편 |
| 8 | 우주인 | 2.7%, 첫 정식 실행에서 통과 | 문서 요청만 |

레퍼런스 이미지는 [generated-images/](generated-images/)에 있다. 블라인드 작업 폴더(`.omc/blind/`)는 용량이 커서 git에서 제외했다.

## 3. 중단 지점: WI-1 (PBR 재질 클래스 + 미리보기 월드)

사용자 피드백(2026-10-23): "재질을 적당히 구분하고 UV를 잘 맞추면 결과물이 훨씬 좋아 보일 것."
Opus 설계 리뷰 결과 전문: [docs/materials-uv-design.md](docs/materials-uv-design.md). WI-1~WI-4 정의, 수용 기준, 위험이 들어 있다.

진단 요약:
1. 스펙에 적은 재질이 텍스처 단계에서 기본값(roughness 0.6, metallic 0)으로 덮어써진다.
2. 미리보기 환경광이 Cycles에 전달되지 않는다. `world.color`만 설정하고 있어서 금속이 검게 나온다.
3. 하드서피스 부품의 텍셀 밀도가 부족하다. 보물상자 부품 24개 중 17개가 레퍼런스 1px당 1텍셀 미만이다(WI-2에서 다룸).
4. 캐릭터 hull UV는 이미 충분하다(1.5~2 텍셀/px). UV를 다시 펴도 효과가 거의 없다.

### WI-1에서 이미 적용된 변경 (검증 안 됨)

executor 에이전트가 2026-10-23 13:25~13:31에 아래 변경을 넣다가 한도에 걸려 멈췄다. 테스트, 스모크, 예제 재생성은 하나도 돌리지 않았다.

| 파일 | 변경 |
|---|---|
| `scripts/bbr/core/pbr.py` (신규) | `CLASSES`(physical/stylized 두 열), `ALIASES`, `resolve`, `column_for`, `entry_for` |
| `scripts/bbr/core/spec.py` | `style.pbr` 기본값 `auto`. 컴포넌트 `material`에 클래스 이름 허용 |
| `schemas/scene_spec.schema.json` | `materials.<key>`에 class, coat 등 키 추가. `style.pbr` 추가 |
| `scripts/bbr/blender/material.py` | `_apply_pbr`, `ensure_material(pbr=)`, `ensure_texture_material(pbr=, maps=)`, emission 연결 |
| `scripts/bbr/blender/hullpost.py` | `assign_texture_material(..., pbr, maps)` |
| `scripts/bbr/blender/partstex.py` | `bake_object`에서 컴포넌트 재질을 해석해 `stats["material"]`에 기록 |
| `scripts/bbr/blender/components.py` | `material_for`가 pbr 값을 해석 |
| `scripts/bbr/blender/render.py` | `preview_world()` 그라디언트 월드 컨텍스트 매니저를 lit preview에 적용 |
| `reconstruction.py`, `core/texelproj.py`, `tests/test_texel_projection.py` | 쓸모없던 `double_face` / `upper_front_conflict` 경고와 테스트 제거 |
| `references/hard-surface-rules.md`, `character-cheatsheet.md`, `SKILL.md`, `THIRD_PARTY_NOTICES.md` | 재질 클래스 표, 바이저 레시피, SKILL.md 규칙 8번, MIT 표기 |
| `examples/treasure_chest`, `examples/mushroom_house` `scene_spec.json` | 재질 클래스 지정 |

현재 상태에서 pytest 결과: **457개 통과, 1개 실패** (아래 7절 참고)

## 4. 다음 할 일 (우선순위 순)

1. **WI-1 마무리와 검증**
   - [ ] `tests/test_scene_spec.py::test_invalid[<lambda>-gold]` 수정: 이제 `gold`가 유효한 별칭(metal_gold)이므로, 테스트에 쓰는 재질 이름을 존재하지 않는 값(예: `unobtanium`)으로 바꾼다
   - [ ] `tests/test_pbr.py` 작성: 프리셋+오버라이드 해석, 별칭, 열 선택, 알 수 없는 클래스 거부
   - [ ] `examples/sd_warrior/scene_spec.json`에 클래스 지정: sword_blade → metal_steel, 방패 등. 몸체 hull은 WI-3 전까지 default 유지
   - [ ] hull 몸체와 face layer가 `material`을 받는지 확인. 우주인 바이저는 `"material": "visor"`
   - [ ] Blender 스모크 테스트 추가
     - (a) 재질 값이 적용되고, `M_ref_texture:` 접두사가 유지되고, `nodes.active.name == "bbr_basecolor"`
     - (b) 클래스 유무에 따라 `render_color` 결과가 `np.array_equal`
     - (c) lit preview 후 `scene.world`가 복원되고 임시 월드가 남지 않음
   - [ ] `RIG_POWER_PER_M2` A/B 재조정(보물상자, 전사)
   - [ ] 보물상자, 버섯 집, 전사 재실행
     - `color_match.per_view.*.error`가 현재 `examples/*/output/report.json`과 같아야 함
     - perspective.png에서 휘도 0.05 미만 픽셀 비율 증가가 2%p 이하
     - robot_character 평균 휘도 변화 ±15% 이내
     - lit 렌더 시간 1.2배 이하
   - [ ] 예제 output 재생성, pytest 전부 통과, `__pycache__` 정리
2. **WI-2**: UV 통계(`core/uvstats.py`)를 리포트에 넣고, 하드서피스 텍셀 예산(`core/uvbudget.py`, `uv_budget` 플래그)을 만든다. 벤치 A/B 후 하드서피스 기본값을 켠다.
3. **WI-3**: 텍셀 단위 재질 클래스. `materials.json` 라벨, `prepare_references.py materials --suggest`, `core/matclass.py`. 전사 갑옷·망토·피부, 우주인 헬멧·슈트를 구분한다.
4. **WI-4**(보류): 뷰 정렬 UV 차트. WI-2 수치에서 p05 < 1.0일 때만 진행한다.
5. 그 다음: 9차 블라인드 테스트로 재질 개선을 확인한다. 범위 밖이지만 효과가 있을 후보는 hull의 안 보이는 면 채움, 비스듬한 뷰 줄무늬, 유령 팔뚝 라벨 게이팅이다.

## 5. 환경 메모 (클라우드에서 이어갈 때)

- 로컬 환경: Windows, Blender 5.1.1 (`C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`), Python 3.13 + numpy, Pillow, pytest
- 클라우드(Linux)에서는 Blender 5.1+를 따로 설치해야 파이프라인, 스모크 테스트, 벤치를 돌릴 수 있다. Blender 없이는 `python3 -m pytest -q tests`만 가능하다.
- pytest 전체 실행에 약 1.5~4분 걸린다.
- 커밋에서 제외한 것(용량 문제):
  - `.omc/`의 scratch, blind, archive (수 GB)
  - `bench/runs/` (약 500MB). `bench/run_bench.py`로 다시 만들 수 있다.
  - `examples/*/output/checkpoints/`
- `.omc/plans/bbr-contracts.md`와 `.omc/research/bbr-roadmap.md`는 초기 설계와 로드맵이다. 커밋에 포함했다.

## 6. 핵심 파일 지도

| 경로 | 역할 |
|---|---|
| `scripts/run_pipeline.py` | 헤드리스 파이프라인 CLI (Blender 안에서 실행) |
| `scripts/prepare_references.py` | 레퍼런스 준비 CLI |
| `scripts/bbr/core/` | 순수 numpy: 스펙, 보정, 마스크, 지표, 라벨, 삼각측량, 피팅, 카빙, 텍셀 투영, pbr |
| `scripts/bbr/blender/` | bpy: 컴포넌트, hull 후처리, 부품 텍스처, 재질, 렌더, 검증 |
| `bench/` | 벤치 하네스, 케이스, 블라인드 프로토콜 |
| `docs/materials-uv-design.md` | 다음 작업(WI-1~4) 설계 전문 |

## 7. 커밋 시점 상태

- pytest(2026-10-25, 로컬 Windows, Python 3.13): **457개 통과, 1개 실패**, 94초
  - 실패: `tests/test_scene_spec.py::test_invalid[<lambda>-gold]`. WI-1에서 `gold`를 재질 별칭으로 추가한 데 따른 예상된 결과다. 코드 회귀가 아니라 테스트를 갱신해야 한다.
  - 이전 459개에서 `double_face` 테스트 1개가 제거됐다. `test_pbr.py`는 아직 없다.
- Blender 스모크 테스트와 벤치는 WI-1 변경 후 아직 돌리지 않았다.
