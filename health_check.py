
"""
Health check and monitoring system for the LLM Factory chatbot.

This module provides comprehensive health checks for all system components
including LLM connectivity, search tools, configuration validation, and
performance monitoring.
"""
import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

try:
    from llm_factory import get_llm, get_llm_embeddings
    from llm_factory.config import validate_environment, LLMConfig
    from tools import get_search_tools
    from utils import InputValidationError, validate_api_key
except ImportError as e:
    logging.warning(f"Import error in health check: {e}")


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    checks: List[HealthCheckResult]
    summary: Dict[str, int]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize health checker.
        
        Args:
            timeout: Timeout for individual health checks in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    async def check_llm_connectivity(self) -> HealthCheckResult:
        """Check LLM connectivity and basic functionality."""
        start_time = time.time()
        
        try:
            # Test LLM creation and basic invocation
            llm = get_llm()
            
            # Simple test prompt
            test_prompt = "Hello, please respond with 'OK'"
            response = await asyncio.wait_for(
                llm.ainvoke([{"role": "user", "content": test_prompt}]),
                timeout=self.timeout
            )
            
            duration = (time.time() - start_time) * 1000
            
            if response and hasattr(response, 'content'):
                return HealthCheckResult(
                    component="llm_connectivity",
                    status="healthy",
                    message="LLM is responsive",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration,
                    details={
                        "response_length": len(str(response.content)),
                        "model": getattr(llm, 'model_name', 'unknown')
                    }
                )
            else:
                return HealthCheckResult(
                    component="llm_connectivity",
                    status="unhealthy",
                    message="LLM response invalid",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration
                )
                
        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="llm_connectivity",
                status="unhealthy",
                message=f"LLM timeout after {self.timeout}s",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="llm_connectivity",
                status="unhealthy",
                message=f"LLM error: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={"error_type": type(e).__name__}
            )
    
    def check_configuration(self) -> HealthCheckResult:
        """Check configuration validity and security."""
        start_time = time.time()
        
        try:
            # Validate environment
            validate_environment()
            
            # Check configuration
            config = LLMConfig.from_env()
            
            # Security checks
            issues = []
            
            # Check API key format
            try:
                validate_api_key(config.api_key)
            except InputValidationError as e:
                issues.append(f"API key issue: {e}")
            
            # Check if using production-ready settings
            if not config.base_url:
                issues.append("No custom base URL configured")
            
            # Check log level
            log_level = os.getenv("LOG_LEVEL", "INFO")
            if log_level == "DEBUG":
                issues.append("Debug logging enabled (not for production)")
            
            duration = (time.time() - start_time) * 1000
            
            if not issues:
                return HealthCheckResult(
                    component="configuration",
                    status="healthy",
                    message="Configuration is valid",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration,
                    details={
                        "model": config.model,
                        "timeout": config.timeout,
                        "has_base_url": bool(config.base_url)
                    }
                )
            else:
                return HealthCheckResult(
                    component="configuration",
                    status="degraded",
                    message=f"Configuration issues: {'; '.join(issues)}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration,
                    details={"issues": issues}
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="configuration",
                status="unhealthy",
                message=f"Configuration error: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={"error_type": type(e).__name__}
            )
    
    async def check_search_tools(self) -> HealthCheckResult:
        """Check search tools availability and functionality."""
        start_time = time.time()
        
        try:
            tools = get_search_tools()
            
            if not tools:
                duration = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    component="search_tools",
                    status="unhealthy",
                    message="No search tools available",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration
                )
            
            # Test a simple search
            test_query = "test health check"
            tool = tools[0]  # Use first available tool
            
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.func, test_query),
                    timeout=self.timeout
                )
                
                duration = (time.time() - start_time) * 1000
                
                if result and isinstance(result, str) and len(result) > 10:
                    return HealthCheckResult(
                        component="search_tools",
                        status="healthy",
                        message=f"Search tools working ({len(tools)} tools)",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        duration_ms=duration,
                        details={
                            "tools_count": len(tools),
                            "primary_tool": tool.name,
                            "result_length": len(result)
                        }
                    )
                else:
                    return HealthCheckResult(
                        component="search_tools",
                        status="degraded",
                        message="Search returned minimal results",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        duration_ms=duration,
                        details={"tools_count": len(tools)}
                    )
                    
            except asyncio.TimeoutError:
                duration = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    component="search_tools",
                    status="degraded",
                    message=f"Search timeout after {self.timeout}s",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration,
                    details={"tools_count": len(tools)}
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="search_tools",
                status="unhealthy",
                message=f"Search tools error: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={"error_type": type(e).__name__}
            )
    
    def check_embeddings(self) -> HealthCheckResult:
        """Check embeddings functionality."""
        start_time = time.time()
        
        try:
            embeddings = get_llm_embeddings()
            
            # Test embedding generation
            test_text = "health check test"
            vectors = embeddings.embed_query(test_text)
            
            duration = (time.time() - start_time) * 1000
            
            if vectors and len(vectors) > 0:
                return HealthCheckResult(
                    component="embeddings",
                    status="healthy",
                    message="Embeddings are working",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration,
                    details={
                        "dimensions": len(vectors),
                        "model": getattr(embeddings, 'model', 'unknown')
                    }
                )
            else:
                return HealthCheckResult(
                    component="embeddings",
                    status="unhealthy",
                    message="Embeddings returned empty result",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="embeddings",
                status="unhealthy",
                message=f"Embeddings error: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={"error_type": type(e).__name__}
            )
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        start_time = time.time()
        
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status based on resource usage
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
            
            status = "healthy"
            if len(issues) > 2:
                status = "unhealthy"
            elif issues:
                status = "degraded"
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=("System resources checked" if not issues
                         else f"Issues: {'; '.join(issues)}"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "available_memory_gb": round(
                        memory.available / (1024**3), 2
                    )
                }
            )
            
        except ImportError:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="system_resources",
                status="degraded",
                message="psutil not available for system monitoring",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="system_resources",
                status="degraded",
                message=f"System check error: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration,
                details={"error_type": type(e).__name__}
            )
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all health checks and return overall system status."""
        checks = []
        
        # Run all checks
        try:
            checks.append(await self.check_llm_connectivity())
        except Exception as e:
            self.logger.error(f"LLM connectivity check failed: {e}")
        
        try:
            checks.append(self.check_configuration())
        except Exception as e:
            self.logger.error(f"Configuration check failed: {e}")
        
        try:
            checks.append(await self.check_search_tools())
        except Exception as e:
            self.logger.error(f"Search tools check failed: {e}")
        
        try:
            checks.append(self.check_embeddings())
        except Exception as e:
            self.logger.error(f"Embeddings check failed: {e}")
        
        try:
            checks.append(self.check_system_resources())
        except Exception as e:
            self.logger.error(f"System resources check failed: {e}")
        
        # Calculate summary
        summary = {
            "healthy": len([c for c in checks if c.status == "healthy"]),
            "degraded": len([c for c in checks if c.status == "degraded"]),
            "unhealthy": len([c for c in checks if c.status == "unhealthy"])
        }
        
        # Determine overall status
        overall_status = "healthy"
        if summary["unhealthy"] > 0:
            overall_status = "unhealthy"
        elif summary["degraded"] > 0:
            overall_status = "degraded"
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=checks,
            summary=summary
        )
    
    def export_health_report(self, health: SystemHealth,
                             format_type: str = "json") -> str:
        """Export health report in specified format."""
        if format_type == "json":
            return json.dumps(asdict(health), indent=2)
        elif format_type == "text":
            lines = [
                f"System Health Report - {health.timestamp}",
                f"Overall Status: {health.status.upper()}",
                "",
                "Summary:",
                f"  Healthy: {health.summary['healthy']}",
                f"  Degraded: {health.summary['degraded']}",
                f"  Unhealthy: {health.summary['unhealthy']}",
                "",
                "Component Details:"
            ]
            
            for check in health.checks:
                lines.append(f"  {check.component}: {check.status.upper()}")
                lines.append(f"    Message: {check.message}")
                lines.append(f"    Duration: {check.duration_ms:.1f}ms")
                if check.details:
                    lines.append(f"    Details: {check.details}")
                lines.append("")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


async def main():
    """Main function for running health checks."""
    # Configure minimal logging for production health checks
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s'
    )
    
    # Suppress third-party library logs
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("langchain").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    
    # Create health checker
    checker = HealthChecker(timeout=10.0)
    
    print("Running system health checks...")
    health = await checker.run_all_checks()
    
    print("\n" + "="*50)
    print("HEALTH CHECK REPORT")
    print("="*50)
    print(checker.export_health_report(health, "text"))
    
    # Exit with appropriate code
    if health.status == "unhealthy":
        exit(1)
    elif health.status == "degraded":
        exit(2)
    else:
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())