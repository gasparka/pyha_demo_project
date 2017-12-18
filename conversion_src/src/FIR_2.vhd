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
    use work.ComplexSfix_0.all;


package FIR_2 is
    type next_t is record
        DELAY: integer;
        TAPS: Typedefs.sfixed0downto_17_list_t(0 to 7);
        acc: Typedefs.sfixed1downto_34_list_t(0 to 7);
        \out\: sfixed(0 downto -17);
    end record;

    type self_t is record
        DELAY: integer;
        TAPS: Typedefs.sfixed0downto_17_list_t(0 to 7);
        acc: Typedefs.sfixed1downto_34_list_t(0 to 7);
        \out\: sfixed(0 downto -17);
        \next\: next_t;
    end record;
    type FIR_2_self_t_list_t is array (natural range <>) of FIR_2.self_t;

    procedure \_pyha_init\(self:inout self_t);

    procedure \_pyha_reset_constants\(self:inout self_t);

    procedure \_pyha_reset\(self:inout self_t);

    procedure \_pyha_deepcopy\(self:inout self_t; other: in self_t);

    procedure \_pyha_list_deepcopy\(self:inout FIR_2_self_t_list_t; other: in FIR_2_self_t_list_t);

    procedure \_pyha_update_registers\(self:inout self_t);

    procedure main(self:inout self_t; x: sfixed(0 downto -17); ret_0:out sfixed(0 downto -17));
end package;

package body FIR_2 is
    procedure \_pyha_init\(self:inout self_t) is
    begin
        self.\next\.DELAY := self.DELAY;
        self.\next\.TAPS := self.TAPS;
        self.\next\.acc := self.acc;
        self.\next\.\out\ := self.\out\;
    end procedure;

    procedure \_pyha_reset_constants\(self:inout self_t) is
    begin
        self.DELAY := 2;
        self.TAPS := (Sfix(-0.040802001953125, 0, -17), Sfix(0.1103363037109375, 0, -17), Sfix(0.21150970458984375, 0, -17), Sfix(0.29248046875, 0, -17), Sfix(0.29248046875, 0, -17), Sfix(0.21150970458984375, 0, -17), Sfix(0.1103363037109375, 0, -17), Sfix(-0.040802001953125, 0, -17));
    end procedure;

    procedure \_pyha_reset\(self:inout self_t) is
    begin
        self.\next\.DELAY := 2;
        self.\next\.TAPS := (Sfix(-0.040802001953125, 0, -17), Sfix(0.1103363037109375, 0, -17), Sfix(0.21150970458984375, 0, -17), Sfix(0.29248046875, 0, -17), Sfix(0.29248046875, 0, -17), Sfix(0.21150970458984375, 0, -17), Sfix(0.1103363037109375, 0, -17), Sfix(-0.040802001953125, 0, -17));
        self.\next\.acc := (Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34), Sfix(0.0, 1, -34));
        self.\next\.\out\ := Sfix(0.0, 0, -17);
        \_pyha_update_registers\(self);
    end procedure;

    procedure \_pyha_deepcopy\(self:inout self_t; other: in self_t) is
    begin
        self.\next\.DELAY := other.DELAY;
        self.\next\.TAPS := other.TAPS;
        self.\next\.acc := other.acc;
        self.\next\.\out\ := other.\out\;
    end procedure;

    procedure \_pyha_list_deepcopy\(self:inout FIR_2_self_t_list_t; other: in FIR_2_self_t_list_t) is
    begin
        for i in self'range loop
            \_pyha_deepcopy\(self(i), other(i));
        end loop;
    end procedure;

    procedure \_pyha_update_registers\(self:inout self_t) is
    begin
        self.acc := self.\next\.acc;
        self.\out\ := self.\next\.\out\;
    end procedure;

    procedure main(self:inout self_t; x: sfixed(0 downto -17); ret_0:out sfixed(0 downto -17)) is
    -- Transposed FIR structure
        variable i: integer;
    begin
        for i in self.acc'range loop
            if i = 0 then
                self.\next\.acc(0) := resize(x * self.TAPS(self.TAPS'length-1), 1, -34, fixed_wrap, fixed_truncate);
            else
                self.\next\.acc(i) := resize(self.acc(i - 1) + x * self.TAPS(self.TAPS'length - 1 - i), 1, -34, fixed_wrap, fixed_truncate);

            end if;
        end loop;
        self.\next\.\out\ := resize(self.acc(self.acc'length-1), 0, -17, fixed_saturate, fixed_truncate);
        ret_0 := self.\out\;
        return;
    end procedure;
end package body;
