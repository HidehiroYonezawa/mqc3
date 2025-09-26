"""Verification of feedforward functions."""

import ast


class _FeedForwardVerifier(ast.NodeVisitor):
    """Verifier class that checks if the body of a function definition is safe for feedforward."""

    def __init__(self) -> None:
        self.local_names = set()
        self.return_stmt_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        allowed_stmt = (ast.Return, ast.Assign, ast.AugAssign, ast.ImportFrom, ast.Expr)

        self._verify_args_of_func_def(node)
        for arg in node.args.args:
            self.local_names.add(arg.arg)
        self._verify_decorator_of_func_def(node)

        if not isinstance(node.body[-1], ast.Return):
            msg = "The function definition must end with a return statement."
            raise ValueError(msg)  # noqa: TRY004

        for stmt in node.body:
            if not isinstance(stmt, allowed_stmt):
                msg = (
                    "Feedforward functions can only include return statements, assignments, "
                    "augmented assignments, import-from statements, and expression statements."
                )
                raise ValueError(msg)  # noqa: TRY004
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None or isinstance(node.value, ast.Tuple):
            msg = "The return value must be a single value and cannot be a tuple or None."
            raise ValueError(msg)
        self.return_stmt_count += 1
        if self.return_stmt_count > 1:
            msg = "Function body must contain exactly one return statement."
            raise ValueError(msg)

        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_names.add(target.id)
            elif isinstance(target, ast.Tuple):  # Unpacking a tuple
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        self.local_names.add(element.id)
                    else:
                        msg = "Assignment targets must be local variables."
                        raise ValueError(msg)  # noqa: TRY004
            else:
                msg = "Assignment targets must be either local variables or unpacked tuples."
                raise ValueError(msg)  # noqa: TRY004

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.local_names.add(node.target.id)
        else:
            msg = "Augmented assignment targets must be local variables."
            raise ValueError(msg)  # noqa: TRY004

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "math":
            msg = "Only the built-in math module is allowed for import."
            raise ValueError(msg)

        for name in node.names:
            if name.name == "*":
                msg = "Importing all (* wildcard) is not allowed."
                raise ValueError(msg)
            self.local_names.add(name.name)

    def visit_Expr(self, node: ast.Expr) -> None:
        allowed_expr = (ast.UnaryOp, ast.BinOp, ast.Call, ast.Constant, ast.Name, ast.NamedExpr)
        if not isinstance(node.value, allowed_expr):
            msg = (
                "Allowed expressions are limited to: "
                "unary operations, binary operations, function calls, constants, names, and named expressions."
            )
            raise ValueError(msg)  # noqa: TRY004
        self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        allowed_unary_op = (ast.USub, ast.UAdd, ast.Invert)
        if not isinstance(node.op, allowed_unary_op):
            msg = "Only the following unary operations are allowed: +, -, ~."
            raise ValueError(msg)  # noqa: TRY004
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        allowed_binary_op = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.LShift,
            ast.RShift,
            ast.BitAnd,
            ast.BitXor,
            ast.BitOr,
            ast.Pow,
            ast.MatMult,
        )
        if not isinstance(node.op, allowed_binary_op):
            msg = "Only the following binary operations are allowed: +, -, *, /, //, %, <<, >>, &, ^, |, **, @."
            raise ValueError(msg)  # noqa: TRY004

        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> None:
        allowed_builtin_func = {"abs", "bool", "complex", "divmod", "float", "int", "pow", "round"}
        if not isinstance(node.func, ast.Name) or (node.func.id not in (allowed_builtin_func | self.local_names)):
            msg = (
                "Only the following functions can be called: abs, bool, complex, divmod, float, int, pow, round, "
                "and functions imported from the math module."
            )
            raise ValueError(msg)

        for arg in node.args:
            self.visit(arg)

        for kwarg in node.keywords:
            self.visit(kwarg.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        pass

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.local_names:
            msg = "Names must be defined within the local scope and cannot refer to external definitions."
            raise ValueError(msg)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            msg = "Assignment targets must be local variables."
            raise ValueError(msg)  # noqa: TRY004
        self.local_names.add(node.target.id)

    def _verify_args_of_func_def(self, func_def: ast.FunctionDef) -> None:
        """Verify that arguments of a function definition is safe for feedforward.

        Args:
            func_def (ast.FunctionDef): A function definition to verify.

        Raises:
            ValueError: If the function is not safe.
        """
        if len(func_def.args.args) != 1:
            msg = "Function must have a single argument."
            raise ValueError(msg)
        if func_def.args.posonlyargs:
            msg = "Function must not have positional-only arguments."
            raise ValueError(msg)
        if func_def.args.vararg:
            msg = "Function must not have variable length arguments."
            raise ValueError(msg)
        if func_def.args.kwonlyargs:
            msg = "Function must not have keyword-only arguments."
            raise ValueError(msg)
        if func_def.args.kw_defaults:
            msg = "Function must not have keyword defaults."
            raise ValueError(msg)
        if func_def.args.defaults:
            msg = "Function must not have defaults."
            raise ValueError(msg)

    def _verify_decorator_of_func_def(self, func_def: ast.FunctionDef) -> None:
        """Verify that decorators of a function definition is safe for feedforward.

        Args:
            func_def (ast.FunctionDef): A function definition to verify.

        Raises:
            ValueError: If the function is not safe.
        """
        if len(func_def.decorator_list) > 1:
            msg = "Function must have at most one decorator."
            raise ValueError(msg)
        if func_def.decorator_list:
            decorator = func_def.decorator_list[0]
            if not isinstance(decorator, ast.Name):
                msg = "Decorator must be a name."
                raise ValueError(msg)
            if decorator.id != "feedforward":
                msg = "Decorator must be 'feedforward'."
                raise ValueError(msg)


def verify_feedforward(f: str) -> None:
    """Verify whether the given function string is safe for use as a feedforward function.

    A feedforward function must satisfy the following constraints:

    - It must define exactly one function.
    - The function must have a single argument (no default values, no `*args`, `**kwargs`, or keyword-only arguments).
    - Only the following statements are allowed: `return`, `assign`, `augassign`, `import from math`, and simple expressions.
    - The function body must end with a single `return` statement.
    - Only a limited set of operators is allowed (`+`, `-`, `~`, `*`, `/`, `//`, `%`, `<<`, `>>`, `&`, `^`, `|`, `**`, `@`).
    - Only specific built-in functions (`abs`, `bool`, `complex`, `divmod`, `float`, `int`, `pow`, `round`) and functions from the `math` module may be used.
    - All variables must be defined locally within the function body.

    Args:
        f (str): The source code string of the function to verify.

    Raises:
        ValueError: If the function violates any of the feedforward constraints.
    """  # noqa: E501
    parsed = ast.parse(f, feature_version=(3, 10))

    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        msg = "Function must be a single function definition."
        raise ValueError(msg)

    verifier = _FeedForwardVerifier()
    verifier.visit(parsed.body[0])
