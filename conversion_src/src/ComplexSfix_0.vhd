-- generated by pyha 0.0.6
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


-- Complex number with ``.real`` and ``.imag`` elements. Default type is ``ComplexSfix(left=0, right=-17)``, Python ``complex`` values will be converte to this.
-- :param val:
-- :param left: left bound for both components
-- :param right: right bound for both components
-- :param overflow_style: 'wrap' (default) or 'saturate'.
-- :param round_style: 'truncate' (default) or 'round'
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
-- >>> ComplexSfix(a, b)
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



    -- internal pyha functions
    procedure pyha_update_registers(self:inout self_t);
    procedure pyha_reset(self:inout self_t);
    procedure pyha_init_next(self:inout self_t);
    procedure pyha_reset_constants(self:inout self_t);
    procedure pyha_deepcopy(self:inout self_t; other: in self_t);
    procedure pyha_list_deepcopy(self:inout ComplexSfix_0_self_t_list_t; other: in ComplexSfix_0_self_t_list_t);
end package;

package body ComplexSfix_0 is


    procedure pyha_reset(self:inout self_t) is
        -- executed on reset signal. Reset values are determined from initial values of Python variables.
    begin
        self.\next\.real := Sfix(0.07321929931640625, 0, -17);
        self.\next\.imag := Sfix(0.36014556884765625, 0, -17);
        pyha_update_registers(self);
    end procedure;

    procedure pyha_update_registers(self:inout self_t) is
        -- loads 'next' values to registers, executed on clock rising edge
    begin
        self.real := self.\next\.real;
        self.imag := self.\next\.imag;
    end procedure;

    procedure pyha_init_next(self:inout self_t) is
        -- sets all .next's to current register values, executed before 'main'. 
        -- thanks to this, '.next' variables are always written before read, so they can never be registers
    begin
        self.\next\.real := self.real;
        self.\next\.imag := self.imag;
    end procedure;

    procedure pyha_reset_constants(self:inout self_t) is
        -- reset CONSTANTS, executed before 'main'. Helps synthesis tools to determine constants.
    begin

    end procedure;

    procedure pyha_deepcopy(self:inout self_t; other: in self_t) is
        -- copy 'other' to 'self.next'. ':=' cannot be used as it would directly copy to 'self'
    begin
        self.\next\.real := other.real;
        self.\next\.imag := other.imag;
    end procedure;

    procedure pyha_list_deepcopy(self:inout ComplexSfix_0_self_t_list_t; other: in ComplexSfix_0_self_t_list_t) is
        -- run deepcopy for each list element
    begin
        for i in self'range loop
            pyha_deepcopy(self(i), other(i));
        end loop;
    end procedure;

end package body;
