# 인수인계 (HANDOFF) — Blender 블루프린트 재구성 스킬

마지막 업데이트: 2026-10-25 (클라우드 세션: WI-1 완료) · 이전 작업: 로컬 Claude Code 세션 `2710d383` (2026-10-22 ~ 2026-10-23 13:31 KST, 주간 사용량 한도에 걸려 중단)

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

## 3. WI-1 (PBR 재질 클래스 + 미리보기 월드): 완료

사용자 피드백(2026-10-23): "재질을 적당히 구분하고 UV를 잘 맞추면 결과물이 훨씬 좋아 보일 것."
설계 전문: [docs/materials-uv-design.md](docs/materials-uv-design.md).

로컬 세션에서 executor가 넣은 변경(`core/pbr.py`, spec/schema, `material.py`, `hullpost.py`, `partstex.py`, `components.py`, `render.preview_world`, 문서)을 클라우드 세션(2026-10-25)에서 검증하고 마무리했다.

| 항목 | 결과 |
|---|---|
| `test_scene_spec` gold 케이스 | 존재하지 않는 `unobtanium`으로 교체 |
| `tests/test_pbr.py` | 15개: 프리셋+오버라이드, 별칭, 열 선택, 알 수 없는 클래스 거부, 스키마 enum과 `class_names()` 일치 |
| `test_partfit` 콘 끝 반지름 | 4px 반지름에 rel 5%는 마스크 양자화보다 작아 Linux에서 실패(0.00427). 반 픽셀 절대 허용오차로 변경 |
| Blender 스모크 `check_materials` | (a) hull(stylized)과 평면 부품(physical)에 클래스 값 적용, `M_ref_texture` 접두사, 활성 노드 `bbr_basecolor` (b) 클래스 유무로 `render_color`가 `np.array_equal` (c) lit preview 뒤 `scene.world` 복원, 임시 월드 없음 → `SMOKE OK` |
| hull 몸체 / face layer | 둘 다 `bake_object`에서 `comp.material`을 해석함을 확인(스모크 a). 바이저는 `"material": "visor"` |
| sd_warrior 클래스 | horn → plastic, pommel → metal_gold, grip/칼집(`sword_blade`, 레퍼런스상 갈색 칼집)/손 → leather, 팔 → metal_steel, shield → wood, face → skin. body hull은 WI-3 전까지 default |
| `RIG_POWER_PER_M2` A/B (12/18/24) | 18 유지. 24는 전사 갑옷이 하얗게 날아가고, 12는 상자가 어두움 |

수용 기준 (Linux, bpy 5.1.1):

| 기준 | 결과 |
|---|---|
| color_match 오차 불변 | 전사를 클래스 있음/없음으로 단독 실행하면 세 뷰 모두 비트 단위로 같음. 커밋된 Windows 결과와는 1e-4~1e-3 차이(로봇처럼 클래스 없는 예제도 같은 폭: 플랫폼 차이) |
| 휘도 0.05 미만 픽셀 증가 ≤ 2%p | 상자 0.67% → 1.06%(+0.39%p), 전사 0.50% → 0.07%, 로봇 0.11% → 0% |
| 로봇 평균 휘도 ±15% | 0.630 → 0.659 (+4.7%) |
| lit 렌더 시간 ≤ 1.2배 | 같은 .blend에서 월드 없음 대비 상자 1.12배, 전사 1.13배 |

발견한 것: 손 같은 부품의 UV `pack_islands(shape_method="CONCAVE")`는 기계 부하에 따라 결과가 달라진다(같은 메시, 병렬 4개 실행 vs 단독 실행). 단독 실행끼리는 결정론적이다. 벤치를 병렬로 돌릴 때 색 오차가 1e-4 수준으로 흔들릴 수 있다.

예제 output(상자, 전사, 로봇, 테이블, 의자)을 Linux에서 다시 만들었다. 버섯 집은 원래 output을 커밋하지 않았다.

## 4. 다음 할 일 (우선순위 순)

1. ~~WI-1 마무리와 검증~~ (완료, 3절)
2. **WI-2**: UV 통계(`core/uvstats.py`)를 리포트에 넣고, 하드서피스 텍셀 예산(`core/uvbudget.py`, `uv_budget` 플래그)을 만든다. 벤치 A/B 후 하드서피스 기본값을 켠다.
3. **WI-3**: 텍셀 단위 재질 클래스. `materials.json` 라벨, `prepare_references.py materials --suggest`, `core/matclass.py`. 전사 갑옷·망토·피부, 우주인 헬멧·슈트를 구분한다.
4. **WI-4**(보류): 뷰 정렬 UV 차트. WI-2 수치에서 p05 < 1.0일 때만 진행한다.
5. 그 다음: 9차 블라인드 테스트로 재질 개선을 확인한다. 범위 밖이지만 효과가 있을 후보는 hull의 안 보이는 면 채움, 비스듬한 뷰 줄무늬, 유령 팔뚝 라벨 게이팅이다.

## 5. 환경 메모 (클라우드에서 이어갈 때)

- 클라우드(Linux) 세션: `.claude/settings.json`의 SessionStart 훅(`.claude/hooks/install_blender.sh`)이 PyPI `bpy==5.1.1`, numpy/Pillow/pytest, EGL 라이브러리를 설치하고 `/usr/local/bin/blender` 래퍼(`.claude/hooks/blender_shim.py`)를 만든다. download.blender.org는 프록시에서 막혀 있고 PyPI만 열려 있어서 이 방식을 쓴다. 래퍼는 `-b`, `--factory-startup`, `-P`, `--python-expr`, `--python-exit-code`, `--`를 지원한다. 예제 하나가 단독 실행으로 4~9분 걸린다.
- 로컬 환경: Windows, Blender 5.1.1 (`C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`), Python 3.13 + numpy, Pillow, pytest
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

- pytest(2026-10-25, 클라우드 Linux, Python 3.13, numpy 2.5.3): **473개 전부 통과**, 약 4분
- Blender 스모크(bpy 5.1.1): `SMOKE OK` (`MATERIALS OK` 포함)
- 벤치(`bench/run_bench.py`)는 WI-1 후 아직 돌리지 않았다.
