# class Calculator:
#     @staticmethod
#     def multiply(a: float, b: float) -> int:
#         """
#         Multiply two integers.

#         Args:
#             a (int): The first integer.
#             b (int): The second integer.

#         Returns:
#             int: The product of a and b.
#         """
#         return a * b
    
#     @staticmethod
#     def calculate_total(*x: float) -> float:
#         """
#         Calculate sum of the given list of numbers

#         Args:
#             x (list): List of floating numbers

#         Returns:
#             float: The sum of numbers in the list x
#         """
#         return sum(x)
    
#     @staticmethod
#     def calculate_daily_budget(total: float, days: int) -> float:
#         """
#         Calculate daily budget

#         Args:
#             total (float): Total cost.
#             days (int): Total number of days

#         Returns:
#             float: Expense for a single day
#         """
#         return total / days if days > 0 else 0
    
    
class Calculator:

    @staticmethod
    def multiply(a: float, b: float) -> float:
        """
        Safely multiply two numbers. Converts inputs to float.
        """
        try:
            a = float(a)
            b = float(b)
        except Exception:
            raise ValueError(f"[Calculator.multiply] Invalid numeric input: a={a}, b={b}")

        return a * b

    @staticmethod
    def calculate_total(*x: float) -> float:
        """
        Calculate sum of numeric values. Converts each to float.
        """
        total = 0.0
        for value in x:
            try:
                total += float(value)
            except Exception:
                raise ValueError(f"[Calculator.calculate_total] Invalid value: {value}")

        return total

    @staticmethod
    def calculate_daily_budget(total: float, days: int) -> float:
        """
        Calculate daily budget with safe conversions.
        """
        try:
            total = float(total)
            days = float(days)
        except Exception:
            raise ValueError(f"[Calculator.calculate_daily_budget] Invalid inputs: total={total}, days={days}")

        return total / days if days > 0 else 0.0
