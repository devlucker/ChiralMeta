"""
ChiralMeta Featurizer — RDKit 1544-dim feature extractor & 2D SVG renderer.
"""

import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.DataStructs import ConvertToNumpyArray
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ----------------------------------------------------------------
# 1544-dim featurizer
# ----------------------------------------------------------------

def _morgan(smiles: str, n_bits: int = 512, radius: int = 2) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    if not HAS_RDKIT or not isinstance(smiles, str) or smiles.strip() in ("", "None", "nan"):
        return arr
    smiles = smiles.replace("~", "-")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, useChirality=True
    )
    ConvertToNumpyArray(fp, arr)
    return arr


def compute_feature_vector(
    reactant_smiles: str,
    ligand_smiles: str,
    solvent_smiles: str,
    pressure: float,
    temperature: float,
    sc_ratio: float,
    metal: str = "Co",
    additive_smiles: str = "None",
) -> np.ndarray:
    """
    Build the exact 1,544-dimensional feature vector:
      0..511   : Morgan FP (radius=2, 512 bits) of Reactant SMILES
      512..1023: Morgan FP of Ligand SMILES
      1024..1535: Morgan FP of Solvent SMILES
      1536..1543: [pressure, temperature, S/C, metal_1(Co), metal_2(Ir), metal_3(Rh), add_1, add_2]
    """
    react_fp = _morgan(reactant_smiles)
    lig_fp   = _morgan(ligand_smiles)
    solv_fp  = _morgan(solvent_smiles)

    metal_upper = str(metal).strip().upper()
    metal_1 = 1.0 if metal_upper == "CO" else 0.0
    metal_2 = 1.0 if metal_upper == "IR" else 0.0
    metal_3 = 1.0 if metal_upper == "RH" else 0.0

    add_present = (
        isinstance(additive_smiles, str)
        and additive_smiles.strip() not in ("", "None", "nan")
    )
    add_1 = 1.0 if add_present else 0.0
    add_2 = 1.0 - add_1

    cond = np.array(
        [pressure, temperature, sc_ratio, metal_1, metal_2, metal_3, add_1, add_2],
        dtype=np.float32,
    )

    return np.concatenate([react_fp, lig_fp, solv_fp, cond]).astype(np.float32)


# ----------------------------------------------------------------
# Molecular descriptors (for UI display cards)
# ----------------------------------------------------------------

def compute_mol_descriptors(smiles: str) -> dict:
    """Return dict of physicochemical descriptors for display."""
    defaults = {
        "formula": "N/A", "mol_weight": "N/A",
        "chiral_centers": 0, "rotatable_bonds": 0,
        "aromatic_rings": 0, "hbd": 0, "hba": 0,
        "tpsa": "N/A", "smiles": smiles or "",
    }
    if not HAS_RDKIT or not isinstance(smiles, str) or smiles.strip() in ("", "None", "nan"):
        return defaults

    smiles = smiles.replace("~", "-")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return defaults

    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral = sum(
            1 for a in mol.GetAtoms()
            if a.GetChiralTag().name != "CHI_UNSPECIFIED"
        )
        return {
            "formula":        rdMolDescriptors.CalcMolFormula(mol),
            "mol_weight":     round(Descriptors.MolWt(mol), 2),
            "chiral_centers": chiral,
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "hbd":            rdMolDescriptors.CalcNumHBD(mol),
            "hba":            rdMolDescriptors.CalcNumHBA(mol),
            "tpsa":           round(Descriptors.TPSA(mol), 2),
            "smiles":         Chem.MolToSmiles(mol, isomericSmiles=True),
        }
    except Exception:
        return defaults


# ----------------------------------------------------------------
# 2D SVG renderer
# ----------------------------------------------------------------

def render_mol_svg(
    smiles: str,
    width: int = 420,
    height: int = 280,
    add_stereo_annotation: bool = True,
    highlight_chiral: bool = True,
) -> str:
    """
    Render a molecule to an SVG string using RDKit's MolDraw2DSVG.
    Returns an inline SVG string or an error placeholder SVG.
    """
    if not HAS_RDKIT:
        return _placeholder_svg(width, height, "RDKit unavailable")

    if not isinstance(smiles, str) or smiles.strip() in ("", "None", "nan", "NOT_PROVIDED"):
        return _placeholder_svg(width, height, "No SMILES")

    smiles = smiles.replace("~", "-")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _placeholder_svg(width, height, f"Invalid SMILES")

    try:
        from rdkit.Chem import Draw
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts   = drawer.drawOptions()
        opts.addStereoAnnotation    = add_stereo_annotation
        opts.addAtomIndices         = False
        opts.bondLineWidth          = 1.8
        opts.padding                = 0.12
        opts.minFontSize            = 12
        opts.multipleBondOffset     = 0.18

        # Highlight chiral atoms (light teal)
        atom_colors = {}
        highlight_atoms = []
        if highlight_chiral:
            from rdkit.Chem.rdchem import ChiralType
            teal = (0.06, 0.714, 0.604, 0.25)   # rgba teal translucent
            for atom in mol.GetAtoms():
                if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                    idx = atom.GetIdx()
                    highlight_atoms.append(idx)
                    atom_colors[idx] = teal

        if highlight_atoms:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
            )
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Fix background for dark theme
        svg = svg.replace(
            "background-color:#ffffff",
            "background-color:transparent"
        ).replace(
            "background:#ffffff",
            "background:transparent"
        )
        return svg
    except Exception as e:
        return _placeholder_svg(width, height, str(e)[:60])


def _placeholder_svg(w: int, h: int, msg: str) -> str:
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
  <rect width='{w}' height='{h}' fill='#1E293B' rx='8'/>
  <text x='{w//2}' y='{h//2-10}' text-anchor='middle' fill='#64748B'
        font-family='Inter,sans-serif' font-size='14'>⚗️ Structure Preview</text>
  <text x='{w//2}' y='{h//2+14}' text-anchor='middle' fill='#475569'
        font-family='JetBrains Mono,monospace' font-size='11'>{msg}</text>
</svg>"""
