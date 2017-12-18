-- generated by pyha 0.0.4
library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.fixed_float_types.all;
    use ieee.fixed_pkg.all;
    use ieee.math_real.all;

library work;
    use work.PyhaUtil.all;
    use work.Typedefs.all;
    use work.all;


-- Complex type with 'real' and 'imag' elements, to access underlying Sfix elements.
-- :param val:
-- :param left: left bound for both components
-- :param right: right bound for both components
-- :param overflow_style: fixed_saturate(default) or fixed_wrap
-- >>> a = ComplexSfix(0.45 + 0.88j, left=0, right=-17)
-- >>> a
-- 0.45+0.88j [0:-17]
-- >>> a.real
-- 0.4499969482421875 [0:-17]
-- >>> a.imag
-- 0.8799972534179688 [0:-17]
-- Another way to construct it:
-- >>> a = Sfix(-0.5, 0, -17)
-- >>> b = Sfix(0.5, 0, -17)
-- >>> _ComplexSfixPy(a, b)
-- -0.50+0.50j [0:-17]
package ComplexSfix_0 is
    type next_t is record
        real: sfixed(0 downto -17);
        imag: sfixed(0 downto -17);
    end record;

    type self_t is record
        real: sfixed(0 downto -17);
        imag: sfixed(0 downto -17);
        \next\: next_t;
    end record;
    type ComplexSfix_0_self_t_list_t is array (natural range <>) of ComplexSfix_0.self_t;

    procedure \_pyha_init\(self:inout self_t);

    procedure \_pyha_reset_constants\(self:inout self_t);

    procedure \_pyha_reset\(self:inout self_t);

    procedure \_pyha_deepcopy\(self:inout self_t; other: in self_t);

    procedure \_pyha_list_deepcopy\(self:inout ComplexSfix_0_self_t_list_t; other: in ComplexSfix_0_self_t_list_t);

    procedure \_pyha_update_registers\(self:inout self_t);

end package;

package body ComplexSfix_0 is
    procedure \_pyha_init\(self:inout self_t) is
    begin
        self.\next\.real := self.real;
        self.\next\.imag := self.imag;
    end procedure;

    procedure \_pyha_reset_constants\(self:inout self_t) is
    begin

    end procedure;

    procedure \_pyha_reset\(self:inout self_t) is
    begin
        self.\next\.real := Sfix(0.097625732421875, 0, -17);
        self.\next\.imag := Sfix(0.480194091796875, 0, -17);
        \_pyha_update_registers\(self);
    end procedure;

    procedure \_pyha_deepcopy\(self:inout self_t; other: in self_t) is
    begin
        self.\next\.real := other.real;
        self.\next\.imag := other.imag;
    end procedure;

    procedure \_pyha_list_deepcopy\(self:inout ComplexSfix_0_self_t_list_t; other: in ComplexSfix_0_self_t_list_t) is
    begin
        for i in self'range loop
            \_pyha_deepcopy\(self(i), other(i));
        end loop;
    end procedure;

    procedure \_pyha_update_registers\(self:inout self_t) is
    begin
        self.real := self.\next\.real;
        self.imag := self.\next\.imag;
    end procedure;


end package body;
