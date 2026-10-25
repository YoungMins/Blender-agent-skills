나는 Blender와 AI Agent를 MCP(Model Context Protocol)로 연결하여 사용하고 있다.

현재 문제는 Agent가 Blender를 직접 조작하거나 매번 Python 코드를 즉석에서 생성하게 하면 모델링 방식과 결과가 매번 달라지고, 제공한 레퍼런스 이미지와 충분히 유사한 결과가 나오지 않는다는 것이다.

이를 해결하기 위해 Claude Code에서 재사용 가능한 **Blender Blueprint / Multi-view Reconstruction Skill**을 만들고 싶다.

이 Skill의 목적은 창작형 3D 생성이 아니다.

핵심 목표는 다음과 같다.

> 정면(Front), 측면(Side), 윗면(Top), 후면(Back) 등의 여러 기준 이미지를 입력받고, 해당 이미지들을 최대한 정확하게 만족하는 3D 모델을 Blender 안에서 재구성한다.

즉 Agent가 자유롭게 “비슷한 물체”를 만드는 것이 아니라, 여러 레퍼런스 뷰를 제약조건으로 사용하여 가능한 한 원본과 동일한 비율, 실루엣, 구조를 가진 3D 모델을 만드는 시스템이어야 한다.

---

# 1. 기본 철학

이 시스템은 다음처럼 동작해야 한다.

User Request

→ Reference Images 분석

→ ViewSpec 생성

→ Shape / SceneSpec 생성

→ Blender에서 Reference Blueprint 배치

→ Base Blockout 생성

→ Major Proportion 보정

→ Secondary Structure 생성

→ Hard Surface Modeling

→ Orthographic Render

→ Reference Image와 Projection 비교

→ Error 분석

→ 수정

→ Validation

→ 최종 모델

중요한 점은 Agent가 Blender 작업 방법을 매번 새로 만들어서는 안 된다는 것이다.

Agent의 역할은 다음과 같다.

- 이미지 분석
- 구조 추론
- 파라미터 결정
- 작업 단계 결정
- 오류 분석
- 수정 판단

실제 Blender 조작은 가능한 한 재사용 가능한 Python 함수와 Skill 내부의 정해진 Workflow를 사용한다.

---

# 2. Skill 이름

다음 이름을 기본으로 사용한다.

`blender-blueprint-reconstruction`

적절한 더 좋은 이름이 있다면 제안할 수 있지만 목적이 분명하게 드러나야 한다.

---

# 3. Skill의 초기 대상

첫 번째 버전에서는 모든 종류의 모델을 처리하려 하지 않는다.

MVP에서는 다음 대상을 우선한다.

- 가구
- 공산품
- 기계형 오브젝트
- 소품
- 하드서피스
- 준하드서피스
- 비교적 명확한 구조를 가진 물체

초기 버전에서는 다음은 우선순위를 낮춘다.

- 사람
- 얼굴
- 동물
- 복잡한 유기체
- 의류
- 천
- 머리카락
- 매우 복잡한 캐릭터

추후 확장 가능하도록 구조는 모듈화한다.

---

# 4. 입력 Reference View

Skill은 가능한 경우 다음 이미지를 사용한다.

- Front
- Side Left 또는 Side Right
- Top
- Back

추가적으로 다음 View도 지원 가능하도록 확장성을 둔다.

- Bottom
- Perspective
- Detail reference
- 3/4 Front
- 3/4 Back

모든 이미지가 항상 존재한다고 가정하면 안 된다.

예:

Front + Side만 있는 경우에도 가능한 범위에서 작업해야 한다.

하지만 데이터가 없는 부분은 임의로 확정하지 말고 다음 개념을 사용한다.

- observed
- inferred
- unknown

즉 실제 이미지에서 확인된 형태와 Agent가 추론한 형태를 구분할 수 있어야 한다.

---

# 5. 핵심 데이터 구조

다음 두 종류의 중간 표현을 설계한다.

## ViewSpec

Reference 이미지에 대한 정보를 저장한다.

예:

```json
{
  "views": {
    "front": {
      "path": "references/front.png",
      "enabled": true,
      "camera_type": "orthographic",
      "scale_px_per_meter": null,
      "center": [0.5, 0.5]
    },

    "side": {
      "path": "references/side.png",
      "enabled": true
    },

    "top": {
      "path": "references/top.png",
      "enabled": true
    },

    "back": {
      "path": "references/back.png",
      "enabled": true
    }
  },

  "known_dimensions": {
    "width": null,
    "depth": null,
    "height": null
  },

  "symmetry": {
    "x": true,
    "y": false,
    "z": false
  }
}
```

실제 구현에서는 JSON Schema를 만들어 validation 가능하게 한다.

---

## SceneSpec / ShapeSpec

3D 구조를 표현한다.

예:

```json
{
  "category": "furniture",

  "object_type": "chair",

  "dimensions": {
    "width": 0.52,
    "depth": 0.55,
    "height": 0.81
  },

  "components": [
    {
      "id": "seat",
      "type": "rounded_box",
      "confidence": "observed"
    },

    {
      "id": "left_leg",
      "type": "rectangular_leg",
      "confidence": "observed"
    },

    {
      "id": "rear_support",
      "type": "custom",
      "confidence": "inferred"
    }
  ],

  "style": {
    "hard_surface": true,
    "bevel": true
  }
}
```

SceneSpec은 Agent와 Blender 실행 코드 사이의 계약 역할을 한다.

Agent가 직접 bpy 코드를 무작정 생성하는 구조는 피한다.

---

# 6. Blender Reference Blueprint System

다음 기능을 구현한다.

`setup_reference_views()`

Reference 이미지를 Blender Scene에 자동으로 배치한다.

권장 좌표계:

Front
→ -Y 방향을 바라보는 Plane

Back
→ +Y 방향

Side
→ ±X

Top
→ -Z

각 이미지 Plane은 다음 규칙을 따라야 한다.

- 이미지 중심 정렬
- Scale 통일
- Origin 통일
- 실제 크기가 있다면 실제 단위와 일치
- Orthographic View 기준
- 작업 중 선택 방지 옵션
- 별도 Collection에 저장

Collection 예:

`REF_BLUEPRINT`

모델은 다음 Collection:

`AI_MODEL`

Validator / Debugging:

`AI_DEBUG`

등으로 나눈다.

---

# 7. 모델링 Workflow

Agent가 임의의 순서로 모델링하지 못하도록 Workflow를 강하게 정의한다.

기본 순서:

STEP 1

Reference images import

STEP 2

Reference alignment

STEP 3

Scale calibration

STEP 4

Bounding volume 생성

STEP 5

Front / Side / Top silhouette 기반 Blockout

STEP 6

Major proportions 조정

STEP 7

Symmetry 적용

STEP 8

Primary forms 생성

STEP 9

Secondary forms 생성

STEP 10

Boolean / Cut / Extrusion 등의 구조 작업

STEP 11

Bevel

STEP 12

Normal cleanup

STEP 13

Reference comparison

STEP 14

Correction

STEP 15

Validation

STEP 16

최종 결과 저장

중요:

디테일 작업 전에 반드시 Blockout 검증을 통과해야 한다.

---

# 8. Hard Surface 기본 규칙

첫 번째 버전에서는 다음 규칙을 권장한다.

가능하면 non-destructive workflow를 사용한다.

Modifier 기본 순서는 상황에 따라 다음을 참고한다.

Mirror

→ Boolean

→ Bevel

→ Weighted Normal 또는 적절한 Normal 처리

Scale이 1,1,1이 아닐 경우 필요한 시점에 Apply Scale 한다.

무조건 bpy.ops를 남발하지 않는다.

가능한 경우 다음 API를 우선한다.

- bpy.data
- bmesh
- Object Data API
- Modifier API

bpy.ops가 필요한 경우 명확한 Context를 보장한다.

---

# 9. Primitive / Component Library

Agent가 모든 Geometry를 매번 처음부터 만들지 않도록 재사용 가능한 Primitive 함수를 만든다.

예:

```text
create_box()

create_rounded_box()

create_panel()

create_cylinder()

create_pipe()

create_leg()

create_frame()

create_plate()

create_knob()

create_handle()
```

더 상위 레벨에서는:

```text
create_furniture_leg()

create_drawer()

create_shelf()

create_backrest()

create_support_frame()
```

같은 Semantic Component도 만들 수 있도록 한다.

단 초기 버전에서는 지나치게 많은 Component를 만들지 말고 범용성이 높은 것부터 구현한다.

---

# 10. 가장 중요한 기능: Projection Validation

이 Skill의 핵심 기능이다.

완성된 모델을 보고 Agent가 단순히

“비슷하게 만들어졌다”

라고 판단하면 안 된다.

모델을 다음 Orthographic Camera로 렌더링한다.

- Front
- Side
- Top
- Back

그리고 Reference와 결과 렌더를 비교한다.

최소한 다음 요소를 비교할 수 있게 설계한다.

- Bounding box
- Silhouette
- Object outline
- Major edge positions
- Width / Height
- Depth
- Component position
- Symmetry

가능하면 다음과 같은 Error Score를 계산한다.

```text
Front Error : 4.3%

Side Error : 7.1%

Top Error : 3.8%

Back Error : 5.2%

Total Error : 5.1%
```

초기 구현에서는 정교한 Computer Vision까지 하지 않아도 된다.

MVP에서는 다음 정도로 시작할 수 있다.

- alpha silhouette 비교
- image difference
- bounding rectangle 비교
- edge difference

향후 OpenCV 등을 연결할 수 있게 구조를 만들어 둔다.

---

# 11. Correction Loop

모델 생성 후 다음 반복 구조를 사용한다.

```text
Build

↓

Render Front / Side / Top / Back

↓

Compare

↓

Find largest mismatch

↓

Determine affected axis / component

↓

Modify

↓

Render again

↓

Validate
```

무한 반복하지 않는다.

기본 최대 반복 횟수:

3회

정도로 설정한다.

설정값으로 변경 가능하게 만든다.

---

# 12. View-Constrained Correction

수정은 어떤 View에서 문제가 발생했는지 기반으로 판단한다.

예:

Front View에서 폭이 너무 넓다.

→ X axis 문제 가능성

Side View에서 깊이가 짧다.

→ Y axis 문제 가능성

Top View silhouette가 다르다.

→ X/Y 평면 Shape 문제

Back View에서 돌출부 위치가 틀리다.

→ Rear Component 문제

한 View를 수정했을 때 다른 View가 악화될 수 있으므로 모든 수정 후 전체 View를 다시 검증한다.

---

# 13. Symmetry

SceneSpec에서 다음 정보를 지원한다.

```text
symmetry_x

symmetry_y

symmetry_z
```

대칭 구조라면 가능한 한 절반 또는 일부만 모델링하고 Mirror를 사용한다.

Agent가 Reference를 보고 임의 판단할 수 있지만 확신이 낮으면 inferred 상태로 기록한다.

---

# 14. Confidence System

각 Component 또는 주요 Shape에 다음 Confidence를 저장할 수 있도록 설계한다.

```text
observed

inferred

unknown
```

예:

정면과 측면에서 직접 확인됨:

`observed`

보이지 않지만 구조상 존재한다고 판단:

`inferred`

데이터 부족:

`unknown`

필요하면 Blender Custom Property에 저장한다.

예:

```python
obj["reconstruction_confidence"] = "observed"
```

---

# 15. Idempotent 실행

Skill을 여러 번 실행해도 Blender Scene이 망가지지 않게 한다.

중요 규칙:

같은 작업 재실행 시

오브젝트가 계속 복제되지 않도록 한다.

각 Agent 생성 Object에는 고유 ID를 저장한다.

예:

```python
obj["agent_id"] = "chair_seat_001"
```

재실행하면:

Create가 아니라

Find → Update

방식을 우선 사용한다.

---

# 16. Checkpoint System

각 주요 단계 전에 Checkpoint를 만들 수 있도록 설계한다.

예:

```text
checkpoint_00_initial

checkpoint_01_blockout

checkpoint_02_primary

checkpoint_03_secondary

checkpoint_04_final
```

실패 시 이전 Checkpoint로 돌아갈 수 있게 한다.

Blend 파일 저장 또는 Scene State 방식 중 적절한 방법을 선택한다.

---

# 17. 사용자 승인 지점

완전 자동화를 목표로 하더라도 다음 단계에서는 사용자가 결과를 확인할 수 있도록 한다.

Stage A

Blockout

Stage B

Primary Structure

Stage C

Final Detail

Agent는 각 단계에서 가능한 경우 Preview Render를 생성한다.

향후 MCP Agent가 해당 이미지를 보고 사용자 피드백을 받거나 자동으로 다음 단계로 넘어갈 수 있도록 구조화한다.

---

# 18. Validation

최종 Validation 함수들을 만든다.

예:

```text
validate_dimensions()

validate_scale()

validate_symmetry()

validate_normals()

validate_non_manifold()

validate_modifiers()

validate_materials()

validate_reference_projection()

validate_scene_structure()
```

최종적으로 하나의 함수가 전체 결과를 반환한다.

예:

```python
validate_reconstruction()
```

결과 예:

```json
{
  "passed": false,

  "geometry": {
    "passed": true
  },

  "reference_match": {
    "passed": false,

    "front_error": 0.032,

    "side_error": 0.087,

    "top_error": 0.041
  },

  "recommendation": "Reduce depth of rear support."
}
```

---

# 19. Python SDK

Skill 내부에 Blender 전용 Python SDK 형태의 구조를 만든다.

예:

```text
scripts/

    scene.py

    reference.py

    primitive.py

    mesh.py

    components.py

    modifiers.py

    material.py

    camera.py

    render.py

    projection.py

    validation.py

    checkpoint.py

    reconstruction.py
```

Agent는 가급적 이 API를 사용해야 한다.

즉 매번 bpy 코드 전체를 새로 생성하는 것은 피한다.

---

# 20. Skill 구조

다음 형태를 기본으로 프로젝트를 설계한다.

```text
blender-blueprint-reconstruction/

    SKILL.md

    README.md

    schemas/

        view_spec.schema.json

        scene_spec.schema.json

    scripts/

        scene.py

        reference.py

        primitive.py

        mesh.py

        modifiers.py

        components.py

        camera.py

        render.py

        projection.py

        validation.py

        checkpoint.py

        reconstruction.py

    references/

        reconstruction-workflow.md

        hard-surface-rules.md

        input-image-guidelines.md

        validation-rules.md

        coordinate-system.md

    examples/

        simple_table/

            view_spec.json

            scene_spec.json

        simple_chair/

            view_spec.json

            scene_spec.json

    tests/

        test_view_spec.py

        test_scene_spec.py

        test_geometry_utils.py
```

필요하다면 구조를 개선해도 된다.

---

# 21. SKILL.md 역할

SKILL.md에 모든 코드를 넣지 않는다.

SKILL.md는 Agent에게 다음을 알려주는 역할을 한다.

- 언제 이 Skill을 사용해야 하는가
- 어떤 입력이 필요한가
- 어떤 Workflow를 따라야 하는가
- 어떤 Python Script를 사용해야 하는가
- 무엇을 절대 해서는 안 되는가
- 언제 Validation을 해야 하는가
- 언제 재시도를 해야 하는가
- 데이터가 부족할 때 어떻게 처리해야 하는가

실제 구현은 scripts 내부에서 담당한다.

---

# 22. Skill의 핵심 행동 규칙

SKILL.md에는 다음 원칙을 강하게 명시한다.

1.

Never freely redesign the object.

Reference fidelity is more important than aesthetics.

2.

Do not add details that are not visible unless structurally necessary.

3.

When adding inferred geometry, label it as inferred.

4.

Always validate the blockout before adding details.

5.

Always compare at least Front and Side views when available.

6.

When Top or Back references exist, they must also be validated.

7.

Never consider the reconstruction finished only because Blender execution succeeded.

8.

A reconstruction is complete only after validation.

9.

Prefer editing existing generated objects over recreating the entire scene.

10.

Prefer deterministic scripts over arbitrary generated Blender Python.

---

# 23. Input Image Guideline 문서

`input-image-guidelines.md` 파일도 만들어라.

좋은 Reference 조건을 명시한다.

예:

가장 이상적인 Reference:

- orthographic에 가까움
- 렌즈 왜곡 적음
- 중앙 정렬
- 동일한 Scale
- 단색 배경
- 물체 전체가 보임
- 잘리지 않음
- 정면/측면이 정확한 90도 관계
- 실제 치수 하나 이상 제공

좋지 않은 Reference:

- 강한 Perspective
- Wide angle
- 물체 일부 Crop
- 여러 물체가 겹침
- 각도가 정확하지 않음
- Front와 Side Scale이 다름

Reference 품질에 따라 Confidence를 계산할 수 있게 향후 확장 가능하도록 한다.

---

# 24. MCP Integration

현재 이 Skill은 Blender MCP 환경에서 사용될 예정이다.

MCP Server의 구체적인 구현은 별도 프로젝트일 수 있으므로 강하게 결합하지 않는다.

다만 다음 Semantic Tool Interface와 쉽게 연결 가능하도록 작성한다.

예:

```text
inspect_scene

setup_reference_views

build_from_scene_spec

update_component

render_reference_views

validate_reconstruction

save_checkpoint

restore_checkpoint
```

특히 하나의

`execute_blender_python`

도구에 모든 것을 의존하지 않는 구조를 목표로 한다.

---

# 25. Geometry Nodes

반복적이고 Parameter 기반인 작업은 향후 Geometry Nodes를 이용할 수 있도록 고려한다.

하지만 첫 번째 MVP에서는 Geometry Nodes 자체를 Agent가 매번 생성하지 않도록 한다.

검증된 Geometry Node Group이 있을 경우 Agent는 Node를 만드는 것이 아니라 Parameter만 수정해야 한다.

---

# 26. Coding Rules

Python 코드 작성 시 다음을 지킨다.

- 함수 단위 분리
- Type Hint 사용
- Docstring 사용
- Blender Context 의존성 최소화
- 명확한 에러 메시지
- Logging
- 가능한 Deterministic behavior
- 재실행 안전성
- Magic number 최소화
- Parameter 기반 설계

Blender가 없는 일반 Python 환경에서도 가능한 JSON Schema 및 일부 Utility Test는 실행 가능하게 분리한다.

---

# 27. 현재 구현 범위

이번 작업에서는 완벽한 자동 3D Reconstruction AI를 구현하려 하지 않는다.

첫 번째 목표는 **Skill Framework + Blender Reconstruction Pipeline의 뼈대**다.

즉 다음이 실제로 동작하면 된다.

1.

Front / Side / Top / Back Reference 경로를 ViewSpec으로 정의

2.

Blender에 Reference Plane 생성

3.

Reference Plane 정렬

4.

Known Dimensions 기반 Scale Calibration

5.

간단한 Primitive / Rounded Box 기반 Object 생성

6.

SceneSpec 기반 모델 생성

7.

Orthographic Camera 자동 생성

8.

Front / Side / Top / Back Preview Render

9.

간단한 Projection Validation

10.

Validation Report 생성

11.

Checkpoint

12.

재실행 안전성

---

# 28. 테스트용 Sample

테스트용 모델로 단순 테이블을 사용한다.

입력 구조 예:

```text
Table

Width = 1200mm

Depth = 600mm

Height = 720mm

Top thickness = 35mm

Legs = 4

Leg size = 50mm x 50mm
```

Front / Side / Top Reference가 있다고 가정한다.

이를 SceneSpec으로 정의하고 Blender에서 생성할 수 있는 예제를 작성한다.

두 번째 Example로 Simple Chair도 만들어라.

---

# 29. 최종적으로 원하는 결과

설명만 하지 말고 현재 Repository 안에 실제 Skill 구조와 파일들을 생성해라.

작업 시작 전 현재 Repository 구조를 확인하고 기존 파일을 불필요하게 덮어쓰지 않는다.

완료 후 다음 내용을 요약해서 알려준다.

1.

생성한 Directory 구조

2.

각 파일의 역할

3.

Skill이 동작하는 전체 Workflow

4.

현재 구현된 기능

5.

아직 Stub 또는 TODO인 기능

6.

Blender MCP에 연결하는 방법

7.

테스트 방법

8.

다음 단계로 가장 먼저 개선해야 할 부분

---

# 30. 가장 중요한 설계 원칙

이 프로젝트의 목표는

“AI가 Blender를 자유롭게 잘 다루게 하는 것”

이 아니다.

목표는

> AI의 자유도를 제한하고, Reference Image라는 명확한 제약조건을 이용하여 반복 가능하고 검증 가능한 3D Reconstruction Pipeline을 만드는 것이다.

따라서 다음 구조를 유지한다.

```text
Reference Images

↓

ViewSpec

↓

SceneSpec

↓

Deterministic Blender Functions

↓

3D Model

↓

Orthographic Projection

↓

Reference Comparison

↓

Correction

↓

Validation
```

이 원칙을 절대 흐리지 말고 전체 Skill을 설계하고 구현해라.

추가적으로 구현 중 더 좋은 Architecture가 발견된다면 개선안을 적용해도 되지만,

다음 조건은 반드시 유지한다.

- Reference fidelity 최우선
- Agent 자유도 최소화
- Deterministic workflow
- Intermediate specification 사용
- Validation 필수
- 재실행 안전
- 단계별 수정 가능
- 향후 Component Library 및 Geometry Nodes 확장 가능

이 요구사항을 기준으로 프로젝트를 분석하고 실제 구현을 시작해라.