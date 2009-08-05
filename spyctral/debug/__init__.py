"""
Debugging module for spyctral
"""

__all__ = []

import debug_tools
import jacobi_poly_debug
import laguerre_debug

def all_tests():
    from debug_tools import ValidationContainer
    tests = ValidationContainer()
    tests.extend(jacobi_poly_debug.driver())
    tests.extend(laguerre_debug.driver())
    
    tests.run_tests()
    return tests
