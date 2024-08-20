################################################################
Dataset Information
################################################################

Dataset Name:
 QL-XII-47 KINOMEscan

Dataset Description:
The KINOMEscan assay platform is based on a competition binding assay that is run for a compound of interest against each of a panel of 317 to 456 kinases. The assay has three components: a kinase-tagged phage, a test compound, and an immobilized ligand that the compound competes with to displace the kinase. The amount of kinase bound to the immobilized ligand is determined using quantitative PCR of the DNA tag.  Results for each kinase are reported as "Percent of control", where the control is DMSO and where a 100% result means no inhibition of kinase binding to the ligand in the presence of the compound, and where low percent results mean strong inhibition.

--Data in Package:
20340.csv

--Metadata in Package:
Small_Molecule_Metadata.txt
Protein_Metadata.txt

################################################################
Center-specific Information
################################################################

Center-specific Name:
HMS_LINCS

Center-specific Dataset ID:
20340

Center-specific Dataset Link:
http://lincs.hms.harvard.edu/db/datasets/20340/

################################################################
Assay Information
################################################################

Assay Protocol:
1. T7 kinase-tagged phage strains are grown in parallel in 24-well or 96-well block in a BL21 derived E. coli host for 90 minutes until lysis.<br />
2. Lysates are centrifuged at 6000 g and filtered with a .2 um filter to remove cell debris.<br />
3. Streptavidin-coated magnetic beads are treated with biotinylated kinase ligands for 30 minutes at RT to generate affinity resin.<br />
4. Liganded beads are blocked with excess biotin and washed with blocking buffer (SeaBlock (Pierce) 1% BSA, .05% Tween 20, 1mM DTT) to remove unbound ligand and reduce nonspecific phage binding.<br />
5. Phage lysates, liganded affinity beads, and test compounds are combined in 1X binding buffer (20% SeaBlock, .17 X PBS, .05% Tween 20, 6 mM DTT) in 96-well plates. The final concentration of test compounds is 1 uM.<br />
6. Assay plates are incubated at RT with shaking for 1 hour.<br />
7. Affinity beads are washed 4 X with wash buffer (1X PBS, .05% Tween 20, 1 mM DTT) to remove unbound phage.<br />
8. Beads resuspended after final wash in elution buffer (1X PBS, .05% Tween 20, 2 mM nonbiotinylated affinity ligand) and incubated for 30 minutes at RT.<br />
9. Quantitative PCR is used to measure the amount of phage in each eluate (which is proportional to the amount of kinase bound). Data is presented as % kinase bound by ligand in presence of 1 uM compound compared to DMSO only control reaction.<br />

Date Updated:
2017-12-22

Date Retrieved from Center:
1/1/-4713

################################################################
Metadata Information
################################################################

Metadata information regarding the entities used in the experiments is included in the accompanied metadata. A metadata file per entity category is included in the package. For example, the metadata for all the cell lines that were used in the dataset are included in the Cell_Lines_Metadata.txt file.
Descriptions for each metadata field can be found here: http://www.lincsproject.org/data/data-standards/
[/generic/datapointFile]
[/generic/reagents_studied]
