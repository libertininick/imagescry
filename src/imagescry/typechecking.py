"""Typechecking utilities."""

from beartype import BeartypeConf, beartype

typechecker = beartype(conf=BeartypeConf(is_pep484_tower=True))
