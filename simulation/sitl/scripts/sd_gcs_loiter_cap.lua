-- Searchdrone: if the ground station has been silent for CAP_MS while the vehicle hovers
-- in a LOITER that the RC transmitter did not choose, fly home. Covers the onboard computer
-- dying during a find hover with the whole radio down (FS_OPTIONS bit 4 keeps LOITER).
-- No parameters: CAP_MS equals sd_safety.constants.FC_LOITER_CAP_S (test_lua_constants_match).
local CAP_MS = 150000          -- = FC_LOITER_CAP_S * 1000
local LOITER, RTL, SMART_RTL = 5, 6, 21
local REASON_RC = 1            -- ModeReason::RC_COMMAND
local acted = false

local function update()
  local seen = gcs:last_seen()                     -- ms since boot; 0 = never heard
  -- last_seen() and millis() are uint32_t userdata: compare as such, never with a number.
  local silent = seen:toint() ~= 0 and (millis() - seen) >= uint32_t(CAP_MS)
  if not silent then acted = false end             -- heard again: may act on the next silence
  if silent and not acted and arming:is_armed()
     and vehicle:get_mode() == LOITER
     and vehicle:get_control_mode_reason() ~= REASON_RC then
    acted = true
    if not vehicle:set_mode(SMART_RTL) then vehicle:set_mode(RTL) end
    gcs:send_text(2, "SD failsafe: no GCS 150 s in LOITER, SmartRTL")
  end
  return update, 200
end

gcs:send_text(6, "SD loiter cap ready")
return update, 1000
