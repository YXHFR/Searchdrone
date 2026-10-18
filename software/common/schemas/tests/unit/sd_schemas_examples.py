"""One valid example of every exported model. Coordinates are around CMAC (ArduPilot SITL).

``test_roundtrip`` fails when a model in ``sd_schemas.__all__`` has no example here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

import sd_schemas as s

T = datetime(2026, 10, 15, 12, 30, tzinfo=UTC)
HOME_LAT, HOME_LON = -35.363261, 149.165230
MISSION = UUID("0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f")
CHARTER = UUID("0192f49e-1000-7000-8000-000000000001")
COMMAND = UUID("0192f4a1-7b90-7a22-b1c3-77aa00bb11cc")
APPROVAL = UUID("0192f49e-2000-7000-8000-000000000002")
OTHER = UUID("0192f49e-3000-7000-8000-000000000003")
VEHICLE = "sim-01"

AREA: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [
            [149.1640, -35.3645],
            [149.1665, -35.3645],
            [149.1665, -35.3620],
            [149.1640, -35.3620],
            [149.1640, -35.3645],
        ]
    ],
}
PIN: dict[str, Any] = {"type": "Point", "coordinates": [149.16523, -35.36146]}


def _ll() -> s.LatLon:
    return s.LatLon(lat=HOME_LAT, lon=HOME_LON)


def _lla() -> s.LatLonAlt:
    return s.LatLonAlt(lat=HOME_LAT, lon=HOME_LON, alt_amsl_m=584.0)


def _charter() -> s.MissionCharter:
    return s.MissionCharter(
        charter_id=CHARTER,
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        version=1,
        area=AREA,
        home=_lla(),
        pilot_position=_ll(),
        allowed_skills=["takeoff", "goto", "survey_sector"],
        sectors=[s.Sector(sector_id="A", name="A", area=AREA, order_idx=0)],
        autonomy_mode=s.AutonomyMode.WITHIN_CHARTER,
        fence_hash="ab12",
        approved_by="operator:demo",
        approved_at=T,
    )


def _gate() -> s.GateDecision:
    return s.GateDecision(
        decision="REJECT",
        reason_codes=["ALT_ABOVE_LIMIT", "ALT_ABOVE_CHARTER"],
        rule_ids=["R-ALT"],
        message_sv="Höjden är över uppdragets gräns och 1 till.",
        details={"alt_m": 200},
        side="ground",
    )


def _latch() -> s.OperatorLatch:
    return s.OperatorLatch(
        vehicle_id=VEHICLE,
        action="stop",
        action_seq=5,
        set_by="operator:demo",
        set_at=T,
        set_mono_ms=1000,
    )


def _command() -> s.Command:
    return s.Command(
        id=COMMAND,
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        skill="goto",
        params={"lat": -35.36146, "lon": 149.16523, "alt_m": 40},
        issuer_kind="brain",
        issuer_ref="brain:ground",
        backend="llm",
        rationale_sv="Flyger till P1.",
        in_reply_to=None,
        charter_version=1,
        based_on=s.StateRef(state_epoch=7, state_seq=1187),
        status=s.CommandStatus.DISPATCHED,
        status_reason=None,
        approval_id=None,
        expires_at=None,
        created_at=T,
        updated_at=T,
    )


def _approval() -> s.Approval:
    return s.Approval(
        id=APPROVAL,
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        kind="charter",
        subject_id=CHARTER,
        subject_version=1,
        status="pending",
        summary_sv="Sök av rutan på 40 m.",
        required_confirmations=[],
        default_autonomy_mode=s.AutonomyMode.WITHIN_CHARTER,
        gate=None,
        requested_by="brain:ground",
        requested_at=T,
        expires_at=None,
        decided_by=None,
        decided_at=None,
        reason=None,
    )


def _charter_row() -> s.Charter:
    return s.Charter(
        id=CHARTER,
        mission_id=MISSION,
        version=1,
        status="approved",
        body=_charter(),
        proposed_by="brain:ground",
        approved_by="operator:demo",
        approved_at=T,
        install_status="installed",
        fence_hash="ab12",
        fence_verified_at=T,
        approval_id=APPROVAL,
        created_at=T,
    )


def _mission() -> s.Mission:
    return s.Mission(
        id=MISSION,
        vehicle_id=VEHICLE,
        title="Uppdrag 2026-10-15 12:30",
        status=s.MissionState.ACTIVE,
        paused_by=None,
        returning_cause=None,
        state_seq=4,
        limits=s.SafetyLimits(),
        active_charter_id=CHARTER,
        autonomy_mode=s.AutonomyMode.WITHIN_CHARTER,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=T,
        activated_at=T,
        landed_at=None,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )


def _snapshot() -> s.StateSnapshot:
    return s.StateSnapshot(
        state_seq=1187,
        state_epoch=7,
        last_transition_state_seq=1100,
        executor_state=s.ExecutorState.TRANSIT,
        hold_reason=None,
        held_by=None,
        control_suspended=False,
        expected_mode="GUIDED",
        flight_mode="GUIDED",
        armed=True,
        position=_lla(),
        alt_rel_home_m=40.0,
        groundspeed_ms=8.0,
        heading_deg=0.0,
        battery_pct=81.0,
        active_command_id=COMMAND,
        active_skill="goto",
        progress_pct=40.0,
        resume_available=False,
        charter_id=CHARTER,
        charter_version=1,
        clearance_available=False,
        preflight_ok=True,
        preflight={"PF1": True},
        limits_active=[],
        stop_latch=False,
        battery_rtl_active=False,
        tick_ms_p95=3.5,
        sector_progress={"A": 0.0},
    )


def _telemetry() -> s.Telemetry:
    return s.Telemetry(
        t=T,
        time_boot_ms=123456,
        lat=HOME_LAT,
        lon=HOME_LON,
        alt_amsl_m=624.0,
        alt_rel_home_m=40.0,
        heading_deg=0.0,
        groundspeed_ms=8.0,
        climb_ms=0.0,
        battery_pct=81.0,
        battery_v=15.8,
        flight_mode="GUIDED",
        armed=True,
        landed_state="in_air",
        gps_fix=3,
        gps_sats=12,
        home=_ll(),
        dist_to_pilot_m=120.0,
    )


def _event() -> s.Event:
    return s.Event(
        id=OTHER,
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        t=T,
        source="ground",
        type="mission.state_changed",
        severity="info",
        payload={"from": "READY", "to": "ACTIVE"},
    )


def _vehicle_summary() -> s.VehicleSummary:
    return s.VehicleSummary(
        connected=True,
        armed=True,
        landed=False,
        airborne=True,
        flight_mode="GUIDED",
        executor_state=s.ExecutorState.TRANSIT,
        lat=HOME_LAT,
        lon=HOME_LON,
        alt_rel_home_m=40.0,
        groundspeed_ms=8.0,
        battery_pct=81.0,
        battery_to_reserve_s=900,
        dist_to_pilot_m=120.0,
        gnss_ok=True,
        link_ok=True,
    )


def _decision_log() -> dict[str, Any]:
    return {
        "id": OTHER,
        "turn_id": OTHER,
        "trigger": "operator_message",
        "backend": "llm",
        "engine": "cloud",
        "model_id": "example/model",
        "ai_config_hash": "abc",
        "prompt_sha256": "0" * 64,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_ms": 900,
        "rounds": 1,
        "repairs": 0,
        "tool_calls": [{"name": "goto"}],
        "command_id": COMMAND,
        "gate_decision": "ALLOW",
        "error": None,
        "prompt_json": None,
    }


def examples() -> dict[type[BaseModel], BaseModel]:
    """One instance per exported model, keyed by its class."""
    items: list[BaseModel] = [
        s.LatLon(lat=HOME_LAT, lon=HOME_LON),
        _lla(),
        s.SafetyLimits(),
        s.LimitCaps(),
        s.AbsoluteLimits(),
        s.Sector(sector_id="A", name="Sektor A", area=AREA, order_idx=0),
        _charter(),
        s.CharterProposal(area=AREA, reason_sv="Sök av rutan vid P1."),
        s.CharterValidationError(code="AREA_BEYOND_RANGE", detail_sv="För långt bort.", limit=500),
        s.CharterValidationResult(ok=True),
        s.SurveyResumePoint(
            command_id=COMMAND,
            waypoints=[_ll()],
            alt_m=40,
            speed_ms=7,
            base_index=0,
            seq_offset=0,
            next_index=0,
            interrupted_at=_ll(),
        ),
        s.GotoResumePoint(command_id=COMMAND, lat=HOME_LAT, lon=HOME_LON, alt_m=40, speed_ms=8),
        s.VehicleState(
            vehicle_id=VEHICLE,
            mono_ms=1,
            fc_connected=True,
            gcs_mavlink_seen_mono_ms=None,
            armed=False,
            fc_mode="LOITER",
            fc_system_status=3,
            landed_state="on_ground",
            lat=HOME_LAT,
            lon=HOME_LON,
            alt_rel_home_m=0.0,
            alt_amsl_m=584.0,
            groundspeed_ms=0.0,
            climb_ms=0.0,
            heading_deg=0.0,
            battery_pct=100.0,
            battery_charge_state=1,
            battery_voltage_v=16.8,
            gps_fix_type=3,
            gps_satellites=12,
            ekf_ok=True,
            home=_lla(),
            mission_seq_current=None,
            pilot_position=_ll(),
            pilot_position_source="charter",
        ),
        _latch(),
        s.StateRef(state_epoch=7, state_seq=1187),
        s.GateRequest(
            command_id=COMMAND,
            mission_id=MISSION,
            vehicle_id=VEHICLE,
            skill="goto",
            params={"lat": -35.36146, "lon": 149.16523, "alt_m": 200},
            issuer_kind="brain",
            issuer_ref="brain:ground",
            charter_version=1,
            based_on=s.StateRef(state_epoch=7, state_seq=1187),
            phase="submit",
        ),
        s.ClearanceState(
            clearance_id=OTHER,
            origin="local",
            mission_id=MISSION,
            vehicle_id=VEHICLE,
            charter_id=CHARTER,
            charter_version=1,
            received_mono_ms=1000,
            valid_for_ms=120_000,
            consumed=False,
        ),
        _gate(),
        # skill parameters
        s.TakeoffParams(alt_m=40),
        s.GotoParams(lat=-35.36146, lon=149.16523, alt_m=40, speed_ms=8),
        s.GotoAiParams(pin="P1", alt_m=40),
        s.SurveySectorParams(sector_id="A", alt_m=40),
        s.SurveySectorAiParams(sector="A"),
        s.FollowPathParams(
            points=[
                s.LatLon(lat=-35.36246, lon=149.16523),
                s.LatLon(lat=-35.36146, lon=149.16583),
            ],
            alt_m=30,
            label="pa-1",
        ),
        s.FollowPathAiParams(path="pa-1", alt_m=30),
        s.HoldParams(reason="Väntar på operatören."),
        s.ResumeParams(),
        s.ReturnHomeParams(),
        s.ReportParams(text_sv="Sektor A klar."),
        s.MarkLocationParams(lat=-35.36146, lon=149.16523, label="M1"),
        s.MarkLocationAiParams(label="Ryggsäck", pin="P1"),
        s.ProposeCharterChangeParams(area=AREA, reason_sv="Första ramen."),
        s.OrbitParams(lat=-35.36146, lon=149.16523, radius_m=30, alt_m=40),
        s.InspectDetectionParams(detection_id=OTHER, alt_m=30, standoff_m=20),
        s.CaptureStillsParams(),
        s.SetCameraTiltParams(),
        s.LandAtParams(lat=-35.36146, lon=149.16523),
        s.FollowTrackParams(track_id=OTHER, standoff_m=20, alt_m=30, max_speed_ms=5),
        s.StopFollowParams(),
        s.FollowMeParams(standoff_m=20, alt_m=30, max_speed_ms=5),
        s.FindMeParams(alt_m=30, hover_offset_m=20),
        s.TargetContext(charter=_charter(), position=_lla(), home=_lla()),
        s.GateTargets(points=[_ll()], alts_m=[40]),
        s.SKILL_REGISTRY.get("goto").public(),
        # SD-link
        s.Envelope(
            msg_id=OTHER,
            kind="Heartbeat",
            sender="ground:main",
            vehicle_id=VEHICLE,
            mission_id=None,
            seq=1,
            sent_utc=T,
            ttl_ms=None,
            reliable=False,
            corr_id=None,
            body={},
        ),
        s.LatchRef(action="stop", action_seq=5),
        s.Hello(
            schema_major=1,
            boot_id=7,
            executor_state=s.ExecutorState.IDLE,
            charter_id=None,
            charter_version=None,
            fc_firmware="ArduCopter V4.7.1",
            operator_latch=None,
            last_action_seq=0,
        ),
        s.HelloAck(schema_major=1, server_epoch=OTHER, operator_latch=None),
        s.Heartbeat(),
        s.CharterInstall(charter=_charter()),
        s.CharterAck(charter_id=CHARTER, charter_version=1, accepted=True),
        s.ApprovalRef(approval_id=APPROVAL, command_hash="ff00"),
        s.SkillCommand(
            command_id=COMMAND,
            skill="goto",
            params={"lat": -35.36146, "lon": 149.16523, "alt_m": 40, "speed_ms": 8},
            issuer="brain:ground",
            rationale="Flyger till P1.",
            charter_version=1,
            based_on=s.StateRef(state_epoch=7, state_seq=1187),
            approval=None,
        ),
        s.CommandCancel(command_id=COMMAND),
        s.Ack(ref_id=COMMAND, stage="rejected", reason_codes=["OUTSIDE_CHARTER"], gate=_gate()),
        _snapshot(),
        s.OnboardEvent(type="executor.state_changed", severity="info", data={"to": "HOLD"}),
        s.OperatorOverrideNotice(
            override_id=OTHER,
            kind="stop",
            action_seq=5,
            issuer_kind="operator",
            operator_id="operator:demo",
        ),
        s.OperatorRelease(action_seq=6, operator_id="operator:demo"),
        s.ClearanceConfirmation(
            clearance_id=OTHER,
            mission_id=MISSION,
            vehicle_id=VEHICLE,
            charter_id=CHARTER,
            charter_version=1,
            valid_for_ms=120_000,
            confirmed_by="operator:demo",
            origin="local",
        ),
        s.PilotPosition(lat=HOME_LAT, lon=HOME_LON, source="charter", accuracy_m=None),
        s.OperatorPosition(lat=HOME_LAT, lon=HOME_LON, accuracy_m=5.0, t=T),
        s.CommanderStatus(connected=True, state="ready"),
        s.ManualTakeover(takeover_id=OTHER, active=True, action_seq=7, operator_id="operator:demo"),
        s.ManualControl(takeover_id=OTHER, input_seq=1, pitch=0.5, roll=0, throttle=0.5, yaw=0),
        # panel API
        s.ApiErrorItem(code="less_than_equal", detail_sv="För högt.", loc=["body", "alt_m"]),
        s.ApiError(
            title="Conflict", status=409, code="CHARTER_NOT_APPROVED", detail_sv="Ingen ram."
        ),
        s.Session(
            role="pilot",
            issuer_ref="operator:demo",
            display_name="Demo",
            auth_mode="local",
            expires_at=None,
        ),
        s.LoginRequest(login_code="abc123"),
        s.MapConfig(pack_url="/tiles/demo.pmtiles", center=_ll(), zoom=15),
        s.VideoConfig(whep_url="/api/v1/video/sim-01/main/whep", hls_url=None),
        s.Features(),
        s.PanelConfig(
            default_vehicle_id=VEHICLE,
            map=None,
            video=None,
            vehicle_kind="sim",
            limits_default=s.SafetyLimits(),
            limits_absolute=s.AbsoluteLimits(),
            features=s.Features(),
        ),
        s.Vehicle(id=VEHICLE, name="Sim 1", kind="sim", mav_sysid=1, connected=True),
        _telemetry(),
        s.LinkStatus(sdlink="up", mavlink="up", onboard_heartbeat_age_ms=200, rtt_ms=12.0),
        s.ManualControlState(
            active=True, takeover_id=OTHER, by="operator:demo", since=T, input_age_ms=50
        ),
        s.VehicleLive(
            vehicle_id=VEHICLE,
            telemetry=_telemetry(),
            snapshot=_snapshot(),
            link=s.LinkStatus(
                sdlink="up", mavlink="up", onboard_heartbeat_age_ms=None, rtt_ms=None
            ),
            latch=None,
            airborne=True,
            in_flight=True,
        ),
        s.ActionRequest(client_action_id=OTHER),
        s.ActionDispatched(action_id=OTHER, action_seq=5, paths={"mavlink": "queued"}),
        s.ActionDispatchedVehicle(
            action_id=OTHER, action_seq=5, paths={"mavlink": "queued"}, vehicle_id=VEHICLE
        ),
        s.ActionDispatchedAll(actions=[]),
        s.ActionOutcome(
            action_id=OTHER,
            vehicle_id=VEHICLE,
            action="stop",
            action_seq=5,
            via="button",
            by="operator:demo",
            result="acked",
            mode_requested="BRAKE",
            mode_confirmed="BRAKE",
            paths={"mavlink": "sent", "sdlink": "sent"},
            detail_sv=None,
            t=T,
        ),
        s.ManualControlInput(
            takeover_id=OTHER, input_seq=1, pitch=0, roll=0, throttle=0.5, yaw=0, source="gamepad"
        ),
        _mission(),
        s.MissionSummary(
            id=MISSION,
            vehicle_id=VEHICLE,
            title="Uppdrag",
            status=s.MissionState.DRAFT,
            created_at=T,
        ),
        s.MissionPatch(title="Övning"),
        s.AutonomyRequest(mode=s.AutonomyMode.APPROVE_EACH),
        s.AiToggleRequest(enabled=False),
        s.MissionEndRequest(reason="Övningen klar."),
        _charter_row(),
        _approval(),
        s.ApprovalDecisionRequest(
            decision="approve", subject_version=1, confirmations=["clear_around_drone"]
        ),
        _command(),
        s.OperatorCommandRequest(
            client_command_id=OTHER, skill="goto", params={"lat": -35.36146, "lon": 149.16523}
        ),
        s.CommandResult(command=_command(), gate=_gate()),
        s.ChatMessage(
            id=OTHER,
            seq=1,
            mission_id=MISSION,
            role="operator",
            author_ref="operator:demo",
            kind="text",
            text="stopp",
            in_reply_to=None,
            refs={},
            author_role="pilot",
            parsed_intent="stop",
            handled_by="ground",
            delivery="delivered",
            created_at=T,
        ),
        s.ChatPost(client_msg_id=OTHER, text="Flyg till P1."),
        s.Annotation(
            id=OTHER,
            mission_id=MISSION,
            kind="pin",
            label="P1",
            geometry=PIN,
            created_by="operator:demo",
            created_at=T,
            area_m2=None,
        ),
        s.AnnotationCreate(kind="pin", label="P1", geometry=PIN),
        s.Clearance(
            id=OTHER,
            mission_id=MISSION,
            vehicle_id=VEHICLE,
            charter_id=CHARTER,
            charter_version=1,
            status="active",
            confirmed_by="operator:demo",
            confirmed_at=T,
            valid_until=T,
        ),
        s.MissionDetail(
            mission=_mission(),
            charter=_charter_row(),
            sectors={"type": "FeatureCollection", "features": []},
            pending_approvals=[_approval()],
            active_command=_command(),
            latch=_latch(),
            clearance=None,
        ),
        _event(),
        s.MissionFile(
            id=OTHER,
            kind="tlog",
            rel_path="flight.tlog",
            bytes=1024,
            sha256=None,
            started_at=T,
            ended_at=None,
        ),
        s.Health(
            ready=True,
            version="0.0.0",
            profile="sim",
            parts={"db": "ok", "mavlink": "ok"},
            test_rig=False,
        ),
        s.PartStatus(state="ok", since=T),
        s.SystemStatus(
            parts={"db": s.PartStatus(state="ok", since=T)},
            internet="unknown",
            operator_present=True,
            brain_engine="local",
            version="0.0.0",
            profile="sim",
        ),
        # commander API
        s.MapPin(id="P1", annotation_id=OTHER, lat=-35.36146, lon=149.16523, note="Stigen"),
        _vehicle_summary(),
        s.Situation(
            mission_id=MISSION,
            vehicle_id=VEHICLE,
            state_seq=4,
            snapshot_state_epoch=7,
            snapshot_state_seq=1187,
            mission_status=s.MissionState.ACTIVE,
            paused_by=None,
            returning_cause=None,
            autonomy_mode=s.AutonomyMode.WITHIN_CHARTER,
            ai_enabled=True,
            limits=s.SafetyLimits(),
            charter=_charter(),
            charter_installed=True,
            fence_verified=True,
            pending_proposal=None,
            pending_approvals=[],
            takeoff_clearance_valid_until=None,
            brain_may_takeoff=False,
            override=None,
            vehicle=_vehicle_summary(),
            active_command=_command(),
            sectors=[{"sector_id": "A", "status": "active", "progress": 0.4}],
            pins=[],
            areas=[],
            home=_ll(),
            pilot=_ll(),
            alerts=[],
            recent_events=[_event()],
        ),
        s.BrainMessage(client_msg_id=OTHER, text="Jag flyger till P1."),
        s.SkillRequest(
            command_id=COMMAND,
            skill="goto",
            params={"lat": -35.36146, "lon": 149.16523, "alt_m": 40},
            rationale_sv="Operatören bad om P1.",
            in_reply_to=None,
            based_on=s.StateRef(state_epoch=7, state_seq=1187),
            charter_version=1,
            backend="llm",
            ai_config_hash="abc",
        ),
        s.SkillSubmitResult(
            command_id=COMMAND,
            decision="ALLOW",
            reason_codes=[],
            message_sv="",
            details={},
            approval_id=None,
            status=s.CommandStatus.DISPATCHED,
        ),
        s.EstimateResult(gate=_gate(), duration_s=30.0, battery_after_pct=78.0),
        s.CharterProposalResult(
            charter_id=CHARTER, approval_id=APPROVAL, validation=s.CharterValidationResult(ok=True)
        ),
        s.MissionMemoryUpdate(
            mission_id=MISSION, summary_sv="", plan_note="", summarized_up_to_seq=0
        ),
        s.MissionMemory(
            mission_id=MISSION,
            summary_sv="",
            plan_note="",
            summarized_up_to_seq=0,
            pending_operator_request=None,
            updated_at=None,
        ),
        s.StickAxes(takeover_id=OTHER, input_seq=1, pitch=0, roll=0, throttle=0.5, yaw=0),
        s.DecisionLog(**_decision_log()),
        s.DecisionLogStored(**_decision_log(), prompt_file=None),
        # decision interface
        s.NoulQuestion(instructions="Vill operatören stoppa drönaren?"),
        s.ChoiceQuestion(
            instructions="Vad vill operatören?", criteria={"stop": "Stoppa", "home": "Flyg hem"}
        ),
        s.ScoreQuestion(instructions="Hur bråttom är det?", max_score=4),
        s.BoxQuestion(instructions="Var är personen?", image_id=OTHER),
        s.DecideRequest(
            request_id=OTHER,
            purpose="stop_check",
            state={"message": "stanna nu"},
            questions={"stop": s.NoulQuestion(instructions="Vill operatören stoppa?")},
        ),
        s.NoulAnswer(p_true=0.97),
        s.ChoiceAnswer(choice="stop", probabilities={"stop": 0.9, "home": 0.1}),
        s.ScoreAnswer(score=1, probabilities=[0.2, 0.7, 0.1]),
        s.BoxAnswer(found=False, bbox=None, confidence=0.1),
        s.DecideResult(
            request_id=OTHER,
            engine="local",
            location="local",
            model_id=None,
            latency_ms=120,
            answers={"stop": s.NoulAnswer(p_true=0.97)},
            created_at=T,
        ),
        s.ChatIntentReport(
            request_id=OTHER,
            message_id=OTHER,
            intent="stop",
            probability=0.92,
            engine="jev",
            location="cloud",
            latency_ms=180,
        ),
    ]
    out: dict[type[BaseModel], BaseModel] = {}
    for item in items:
        assert type(item) not in out, f"two examples for {type(item).__name__}"
        out[type(item)] = item
    return out
