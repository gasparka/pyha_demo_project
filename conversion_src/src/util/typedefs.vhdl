-- generated by pyha 0.0.4
library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.fixed_float_types.all;
    use ieee.fixed_pkg.all;
    use ieee.math_real.all;

library work;
    use work.PyhaUtil.all;
    use work.all;

package Typedefs is
    type sfixed0downto_17_list_t is array (natural range <>) of sfixed(0 downto -17);
    type sfixed1downto_34_list_t is array (natural range <>) of sfixed(1 downto -34);
end package;
