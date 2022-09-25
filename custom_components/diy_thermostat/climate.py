"""Adds support for diy thermostat units."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import voluptuous as vol

from homeassistant.components.climate import PLATFORM_SCHEMA, ClimateEntity
from homeassistant.components.climate.const import (
    ATTR_PRESET_MODE,
    PRESET_ACTIVITY,
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_HOME,
    PRESET_NONE,
    PRESET_SLEEP,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import DOMAIN as HA_DOMAIN, CoreState, HomeAssistant, callback
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "DIY Thermostat"

CONF_HEATER = "heater"
CONF_COOLER = "cooler"
CONF_FAN = "fan"
CONF_SENSOR = "target_sensor"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "target_temp_step"

CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_COMFORT,
        PRESET_HOME,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
    )
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Required(CONF_SENSOR): cv.entity_id,
        vol.Optional(CONF_HEATER): cv.entity_id,
        vol.Optional(CONF_COOLER): cv.entity_id,
        vol.Optional(CONF_FAN): cv.entity_id,
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
        vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_KEEP_ALIVE): cv.positive_time_period,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVACMode.COOL, HVACMode.HEAT, HVACMode.HEAT_COOL, HVACMode.FAN_ONLY, HVACMode.OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_TEMP_STEP): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
    }
).extend({vol.Optional(v): vol.Coerce(float) for (k, v) in CONF_PRESETS.items()})


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the generic thermostat platform."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name = config.get(CONF_NAME)
    unique_id = config.get(CONF_UNIQUE_ID)
    heater_entity_id = config.get(CONF_HEATER)
    cooler_entity_id = config.get(CONF_COOLER)
    fan_entity_id = config.get(CONF_FAN)
    sensor_entity_id = config.get(CONF_SENSOR)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)
    target_temp = config.get(CONF_TARGET_TEMP)
    min_cycle_duration = config.get(CONF_MIN_DUR)
    cold_tolerance = config.get(CONF_COLD_TOLERANCE)
    hot_tolerance = config.get(CONF_HOT_TOLERANCE)
    keep_alive = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    precision = config.get(CONF_PRECISION)
    target_temperature_step = config.get(CONF_TEMP_STEP)
    presets = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    unit = hass.config.units.temperature_unit

    async_add_entities(
        [
            DIYThermostat(
                name,
                heater_entity_id,
                cooler_entity_id,
                fan_entity_id,
                sensor_entity_id,
                min_temp,
                max_temp,
                target_temp,
                min_cycle_duration,
                cold_tolerance,
                hot_tolerance,
                keep_alive,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit,
                unique_id,
            )
        ]
    )


class DIYThermostat(ClimateEntity, RestoreEntity):
    """Representation of a DIY Thermostat device."""

    _attr_should_poll = False

    def __init__(
        self,
        name,
        heater_entity_id,
        cooler_entity_id,
        fan_entity_id,
        sensor_entity_id,
        min_temp,
        max_temp,
        target_temp,
        min_cycle_duration,
        cold_tolerance,
        hot_tolerance,
        keep_alive,
        initial_hvac_mode,
        presets,
        precision,
        target_temperature_step,
        unit,
        unique_id,
    ):
        """Initialize the thermostat."""
        self._attr_name = name
        self.heater_entity_id = heater_entity_id
        self.cooler_entity_id = cooler_entity_id
        self.fan_entity_id = fan_entity_id
        self.sensor_entity_id = sensor_entity_id
        self.min_cycle_duration = min_cycle_duration
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._keep_alive = keep_alive
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or next(iter(presets.values()), None)
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        self._attr_hvac_modes = [HVACMode.COOL, HVACMode.HEAT, HVACMode.HEAT_COOL, HVACMode.FAN_ONLY, HVACMode.OFF]
        self._active = False
        self._cur_temp = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE] + list(presets.keys())
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets = presets

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_entity_id], self._async_sensor_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.heater_entity_id], self._async_switch_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.cooler_entity_id], self._async_switch_changed
            )
        )

        if self._keep_alive:
            self.async_on_remove(
                async_track_time_interval(
                    self.hass, self._async_control_heating, self._keep_alive
                )
            )

        @callback
        def _async_startup(*_):
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()
            switch_state = self.hass.states.get(self.heater_entity_id)
            if switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_heater_initial_state())
            cooler_switch_state = self.hass.states.get(self.cooler_entity_id)
            if cooler_switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_cooler_initial_state())
            fan_switch_state = self.hass.states.get(self.fan_entity_id)
            if fan_switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_fan_initial_state())

        if self.hass.state == CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    self._target_temp = self._saved_target_temp
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                self.preset_modes
                and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = old_state.state

        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                self._target_temp = self._saved_target_temp
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

    @property
    def precision(self):
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self):
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision

    @property
    def current_temperature(self):
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def hvac_mode(self):
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self):
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._is_cooler_active:
            return HVACAction.COOLING
        if self._is_heater_active:
            return HVACAction.HEATING
        if self._is_fan_active:
            return HVACAction.FAN
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        return HVACAction.IDLE

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.OFF:
            self._hvac_mode = hvac_mode
            if self._is_device_active:
                await self._async_heater_turn_off()
                await self._async_cooler_turn_off()
                await self._async_fan_turn_off()
        elif hvac_mode == HVACMode.HEAT:
            self._hvac_mode = hvac_mode
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = hvac_mode
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.HEAT_COOL:
            self.hvac_mode = hvac_mode
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.FAN_ONLY:
            self._hvac_mode = hvac_mode
            await self._async_control_heating(force=True)
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        await self._async_control_heating(force=True)
        self.async_write_ha_state()

    @property
    def min_temp(self):
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self):
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    async def _async_sensor_changed(self, event):
        """Handle temperature changes."""
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_heater_initial_state(self):
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_heater_active:
            _LOGGER.warning(
                "The climate mode is OFF, but the switch device is ON. Turning off device %s",
                self.heater_entity_id,
            )
            await self._async_heater_turn_off()

    async def _check_switch_cooler_initial_state(self):
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_cooler_active:
            _LOGGER.warning(
                "The climate mode is OFF, but the switch device is ON. Turning off device %s",
                self.cooler_entity_id,
            )
            await self._async_cooler_turn_off()
    
    async def _check_switch_fan_initial_state(self):
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_fan_active:
            _LOGGER.warning(
                "The climate mode is OFF, but the switch device is ON. Turning off device %s",
                self.fan_entity_id,
            )
            await self._async_fan_turn_off()
    
    @callback
    def _async_switch_changed(self, event):
        """Handle heater switch state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")
        if new_state is None:
            return
        if old_state is None:
            self.hass.create_task(self._check_switch_initial_state())
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state):
        """Update thermostat with latest state from sensor."""
        try:
            cur_temp = float(state.state)
            if math.isnan(cur_temp) or math.isinf(cur_temp):
                raise ValueError(f"Sensor has illegal state {state.state}")
            self._cur_temp = cur_temp
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    async def _async_control_heating(self, time=None, force=False):
        """Check if we need to turn heating on or off."""
        async with self._temp_lock:
            if not self._active and None not in (
                self._cur_temp,
                self._target_temp,
            ):
                self._active = True
                _LOGGER.info(
                    "Obtained current and target temperature. "
                    "Generic thermostat active. %s, %s",
                    self._cur_temp,
                    self._target_temp,
                )

            if not self._active or self._hvac_mode == HVACMode.OFF:
                return

            # If the `force` argument is True, we
            # ignore `min_cycle_duration`.
            # If the `time` argument is not none, we were invoked for
            # keep-alive purposes, and `min_cycle_duration` is irrelevant.
            if not force and time is None and self.min_cycle_duration:
                if self._is_device_active:
                    current_state = STATE_ON
                else:
                    current_state = HVACMode.OFF
                try:
                    long_enough = condition.state(
                        self.hass,
                        self.heater_entity_id,
                        current_state,
                        self.min_cycle_duration,
                    )
                except ConditionError:
                    long_enough = False

                if not long_enough:
                    return

            too_cold = self._target_temp >= self._cur_temp + self._cold_tolerance
            too_hot = self._cur_temp >= self._target_temp + self._hot_tolerance
            if self._is_device_active:
                if self._is_heater_active and not too_cold:
                    _LOGGER.info("Turning off heater %s", self.heater_entity_id)
                    await self._async_heater_turn_off()
                    await self._async_fan_turn_off()
                elif self._is_cooler_active and not too_hot:
                    _LOGGER.info("Turning off cooler %s", self.cooler_entity_id)
                    await self._async_cooler_turn_off()
                    await self._async_fan_turn_off()
            else:
                if too_hot and self._hvac_mode in (HVACMode.COOL, HVACMode.HEAT_COOL):
                    _LOGGER.info("Turning on cooler %s", self.cooler_entity_id)
                    await self._async_fan_turn_on()
                    await self._async_cooler_turn_on()
                elif too_cold and self._hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL):
                    _LOGGER.info("Turning on heater %s", self.heater_entity_id)
                    await self._async_fan_turn_on()
                    await self._async_heater_turn_on()
                elif self._hvac_mode == HVACMode.FAN_ONLY:
                    _LOGGER.info("Turning on fan %s", self.fan_entity_id)
                    await self._async_fan_turn_on()

    @property
    def _is_device_active(self):
        return self._is_heater_active or self._is_cooler_active or self._is_fan_active
    
    @property
    def _is_heater_active(self):
        """If the heater device is currently active."""
        if not self.hass.states.get(self.heater_entity_id):
            return None

        return self.hass.states.is_state(self.heater_entity_id, STATE_ON)
    
    @property
    def _is_cooler_active(self):
        """If the cooler device is currently active."""
        if not self.hass.states.get(self.cooler_entity_id):
            return None

        return self.hass.states.is_state(self.cooler_entity_id, STATE_ON)
    
    @property
    def _is_fan_active(self):
        """If the fan device is currently active."""
        if not self.hass.states.get(self.fan_entity_id):
            return None
        
        return self.hass.states.is_state(self.fan_entity_id, STATE_ON)

    async def _async_heater_turn_on(self):
        """Turn heater toggleable device on."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_ON, data, context=self._context
        )

    async def _async_heater_turn_off(self):
        """Turn heater toggleable device off."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_OFF, data, context=self._context
        )
    
    async def _async_cooler_turn_on(self):
        """Turn cooler toggleable device on."""
        data = {ATTR_ENTITY_ID: self.cooler_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_ON, data, context=self._context
        )
    
    async def _async_cooler_turn_off(self):
        """Turn cooler toggleable device off."""
        data = {ATTR_ENTITY_ID: self.cooler_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_OFF, data, context=self._context
        )
    
    async def _async_fan_turn_on(self):
        """Turn fan toggleable device on."""
        data = {ATTR_ENTITY_ID: self.fan_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_ON, data, context=self._context
        )
    
    async def _async_fan_turn_off(self):
        """Turn fan toggleable device off."""
        data = {ATTR_ENTITY_ID: self.fan_entity_id}
        await self.hass.services.async_call(
            HA_DOMAIN, SERVICE_TURN_OFF, data, context=self._context
        )

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode."""
        if preset_mode not in (self.preset_modes or []):
            raise ValueError(
                f"Got unsupported preset_mode {preset_mode}. Must be one of {self.preset_modes}"
            )
        if preset_mode == self._attr_preset_mode:
            # I don't think we need to call async_write_ha_state if we didn't change the state
            return
        if preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control_heating(force=True)
        else:
            if self._attr_preset_mode == PRESET_NONE:
                self._saved_target_temp = self._target_temp
            self._attr_preset_mode = preset_mode
            self._target_temp = self._presets[preset_mode]
            await self._async_control_heating(force=True)

        self.async_write_ha_state()
