# ceaf_core/utils/context_utils.py

import logging
from typing import Optional, Any, cast

from google.adk.tools import ToolContext

# Attempt to import MBSMemoryService for type checking, but allow it to fail gracefully
# as tools might be used in environments where the full service isn't directly part of the context chain.
try:
    from ..services.mbs_memory_service import MBSMemoryService

    MBS_SERVICE_TYPE_FOR_CHECK = MBSMemoryService
except ImportError:
    MBSMemoryService = None  # type: ignore
    MBS_SERVICE_TYPE_FOR_CHECK = type(None)  # Fallback type for isinstance check
    logging.warning(
        "ContextUtils: MBSMemoryService type not available for precise type checking in _get_mbs_from_context.")

logger = logging.getLogger(__name__)


def get_mbs_from_context(tool_context: ToolContext) -> Optional[
    Any]:  # Return Any to avoid import issues if MBSMemoryService is a placeholder
    """
    Retrieves the MBSMemoryService instance from the ADK ToolContext.
    This function provides a centralized way for tools to access the memory service.
    It tries multiple common ways the service might be available in the context.

    Args:
        tool_context: The ADK ToolContext provided to the tool.

    Returns:
        An instance of MBSMemoryService if found, otherwise None.
    """
    if tool_context is None:
        logger.error("ContextUtils: _get_mbs_from_context received tool_context as None!")
        return None

    memory_service_candidate: Any = None
    ic = None  # InvocationContext

    if hasattr(tool_context, 'invocation_context') and tool_context.invocation_context is not None:
        ic = tool_context.invocation_context
        logger.debug(f"ContextUtils: Found tool_context.invocation_context: {type(ic)}")

        # Standard ADK runner services access
        if hasattr(ic, 'runner') and hasattr(ic.runner, '_services') and isinstance(ic.runner._services, dict):
            memory_service_candidate = ic.runner._services.get('memory_service')
            if memory_service_candidate:
                logger.debug("ContextUtils: Found memory_service_candidate via ic.runner._services")

        # Direct access on runner
        if not memory_service_candidate and hasattr(ic, 'runner') and hasattr(ic.runner, 'memory_service'):
            memory_service_candidate = ic.runner.memory_service
            if memory_service_candidate:
                logger.debug("ContextUtils: Found memory_service_candidate via ic.runner.memory_service")

        # Direct access on invocation_context (e.g., if manually set in some testing scenarios)
        if not memory_service_candidate and hasattr(ic, 'memory_service'):
            memory_service_candidate = ic.memory_service
            if memory_service_candidate:
                logger.debug(
                    "ContextUtils: Found memory_service_candidate via ic.memory_service (direct on invocation_context)")

        # Check if services are stored as a dict directly on invocation_context
        if not memory_service_candidate and hasattr(ic, 'services') and isinstance(ic.services, dict):
            memory_service_candidate = ic.services.get('memory_service')
            if memory_service_candidate:
                logger.debug(
                    "ContextUtils: Found memory_service_candidate via ic.services (dict on invocation_context)")
    else:
        logger.warning(
            "ContextUtils: ToolContext has no 'invocation_context' or it is None. Attempting global fallback.")

    # Global fallback (e.g., for startup tasks or contexts outside full ADK runner)
    if not memory_service_candidate:
        try:
            # This is a conceptual import path; actual project structure determines this.
            # It assumes adk_components is a globally accessible dict in your main FastAPI app module.
            from main import adk_components as main_adk_components
            memory_service_candidate = main_adk_components.get('memory_service')
            if memory_service_candidate:
                logger.info(
                    "ContextUtils: Retrieved memory_service_candidate from main.adk_components (global fallback)")
        except ImportError:
            logger.debug("ContextUtils: main.adk_components not found or importable for global fallback.")
        except AttributeError:
            logger.debug("ContextUtils: main.adk_components found but no 'memory_service' key.")

    if not memory_service_candidate:
        logger.error("ContextUtils: MBSMemoryService instance completely not found in context or fallbacks.")
        return None

    # Perform type check if real MBS class was imported, otherwise duck-type
    if MBS_SERVICE_TYPE_FOR_CHECK is not type(None) and isinstance(memory_service_candidate,
                                                                   MBS_SERVICE_TYPE_FOR_CHECK):  # type: ignore
        logger.info("ContextUtils: Validated memory service instance against imported MBSMemoryService type.")
        return cast(MBSMemoryService, memory_service_candidate)  # type: ignore
    elif (hasattr(memory_service_candidate, 'search_raw_memories') and
          hasattr(memory_service_candidate, 'add_specific_memory') and
          hasattr(memory_service_candidate, 'get_memory_by_id')):
        logger.warning(
            f"ContextUtils: Using DUCK-TYPING for memory service (type: {type(memory_service_candidate)}) "
            "as direct MBSMemoryService type check failed or type was not available."
        )
        return memory_service_candidate  # Return as Any
    else:
        logger.error(f"ContextUtils: Found memory_service_candidate (type: {type(memory_service_candidate)}) "
                     "but it doesn't match expected MBSMemoryService interface (duck-typing failed).")
        return None