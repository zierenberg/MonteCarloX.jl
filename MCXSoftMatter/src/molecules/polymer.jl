struct Polymer{TBond, TBend} <: AbstractMolecule
    offset::Int
    length::Int
    bond::TBond
    bend::TBend
end
