// ============================================================
// Rill.g4 — ANTLR4 grammar for the RILL language.
// Mirrors rill.lark (the executable reference grammar).
// Generate with:  antlr4 -Dlanguage=Python3 Rill.g4
// ============================================================
grammar Rill;

program : statement* EOF ;

statement
    : bind
    | emitStmt
    | whenStmt
    | sustainStmt
    | walkStmt
    | forgeStmt
    | giveStmt
    | haltStmt
    | skipStmt
    | exprStmt
    ;

bind       : expr '->' NAME ;          // rightward binding
emitStmt   : 'emit' expr ;
giveStmt   : 'give' expr? ;
haltStmt   : 'halt' ;
skipStmt   : 'skip' ;
exprStmt   : expr ;

block      : '{' statement* '}' ;

whenStmt   : 'when' expr block elsewhen* otherwise? ;
elsewhen   : 'elsewhen' expr block ;
otherwise  : 'otherwise' block ;

sustainStmt: 'sustain' expr block ;
walkStmt   : 'walk' NAME 'across' expr block ;

forgeStmt  : 'forge' NAME '(' params? ')' block ;
params     : NAME (',' NAME)* ;

// Precedence climbs top-to-bottom; ANTLR resolves with rule order + left-recursion.
expr
    : expr 'either' expr                          # OrExpr
    | expr 'both' expr                            # AndExpr
    | 'flip' expr                                 # FlipExpr
    | expr op=('='|'!='|'<'|'>'|'<='|'>=') expr   # CmpExpr
    | expr op=('*'|'/'|'%') expr                  # MulExpr
    | expr op=('+'|'-') expr                      # AddExpr
    | '-' expr                                    # NegExpr
    | expr '(' args? ')'                          # CallExpr
    | expr '@' expr                               # IndexExpr
    | atom                                        # AtomExpr
    ;

args  : expr (',' expr)* ;

atom
    : INT
    | FLOAT
    | STRING
    | 'yes'
    | 'no'
    | 'void'
    | list
    | NAME
    | '(' expr ')'
    ;

list  : '[' (expr (',' expr)*)? ']' ;

// ---- lexer ----
INT     : [0-9]+ ;
FLOAT   : [0-9]+ '.' [0-9]+ ;
STRING  : '"' ( '\\' . | ~["\\] )* '"' ;
NAME    : [a-zA-Z_][a-zA-Z_0-9]* ;

COMMENT : '~' ~[\r\n]* -> skip ;
WS      : [ \t\r\n]+   -> skip ;
