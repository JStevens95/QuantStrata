"""
Unit tests for deep_hedging.agents.delta module.
"""

import numpy as np
import pytest

from src.deep_hedging.core.types import HedgingState
from src.deep_hedging.agents.delta import DeltaHedgingAgent, NoHedgingAgent


class TestDeltaHedgingAgent:
    """Tests for DeltaHedgingAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create default agent."""
        return DeltaHedgingAgent()
    
    @pytest.fixture
    def state(self):
        """Create test state."""
        return HedgingState(
            spot=100.0,
            time=0.1,
            time_to_maturity=0.15,
            position=0.4,
            pnl=1.0,
            step=10,
            delta_bs=0.55,
            gamma_bs=0.02,
            vega_bs=0.3,
            strike=100.0,
        )
    
    def test_select_action_returns_delta(self, agent, state):
        """Test that action is the BSM delta."""
        action = agent.select_action(state)
        
        assert action == pytest.approx(0.55, rel=1e-6)
    
    def test_select_action_with_scaling(self, state):
        """Test delta scaling."""
        agent = DeltaHedgingAgent(delta_scaling=0.5)
        action = agent.select_action(state)
        
        assert action == pytest.approx(0.275, rel=1e-6)  # 0.55 * 0.5
    
    def test_select_action_with_clipping(self, state):
        """Test delta clipping."""
        agent = DeltaHedgingAgent(clip_delta=(0.0, 0.5))
        action = agent.select_action(state)
        
        assert action == 0.5  # Clipped from 0.55
    
    def test_select_action_from_dict(self, agent):
        """Test action from dict state."""
        state_dict = {
            "delta_bs": 0.6,
            "gamma_bs": 0.01,
            "spot": 100.0,
        }
        
        action = agent.select_action(state_dict)
        assert action == pytest.approx(0.6, rel=1e-6)
    
    def test_update_is_noop(self, agent):
        """Test that update returns None (no learning)."""
        result = agent.update(transitions=[])
        assert result is None
    
    def test_get_set_parameters(self, agent):
        """Test parameter serialisation."""
        params = agent.get_parameters()
        
        assert "delta_scaling" in params
        assert params["delta_scaling"] == 1.0
        
        # Modify and restore
        agent.set_parameters({"delta_scaling": 0.8})
        assert agent.delta_scaling == 0.8
    
    def test_name_property(self, agent):
        """Test name property."""
        assert agent.name == "DeltaHedging"


class TestNoHedgingAgent:
    """Tests for NoHedgingAgent."""
    
    def test_select_action_always_zero(self):
        """Test that action is always zero."""
        agent = NoHedgingAgent()
        
        # Any state should give zero
        action = agent.select_action({"spot": 100, "delta_bs": 0.5})
        assert action == 0.0
    
    def test_name_property(self):
        """Test name property."""
        agent = NoHedgingAgent()
        assert agent.name == "NoHedging"
