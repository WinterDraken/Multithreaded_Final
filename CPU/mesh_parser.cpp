#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include "mesh_parser.h"

// Gmsh v2 Parser 
void parseMSHv2(std::ifstream &file, std::vector<Node> &nodes, std::vector<Element> &elements) {
    std::string line;
    bool readingNodes = false, readingElements = false;

    while (std::getline(file, line)) {
        if (line == "$Nodes") { readingNodes = true; continue; }
        if (line == "$EndNodes") { readingNodes = false; continue; }
        if (line == "$Elements") { readingElements = true; continue; }
        if (line == "$EndElements") { readingElements = false; continue; }

        if (readingNodes) {
            std::istringstream iss(line);
            int id; double x, y, z;
            if (iss >> id >> x >> y >> z)
                nodes.push_back({id, x, y, z});
        }

        if (readingElements) {
            std::istringstream iss(line);
            Element e;
            int numTags;
            if (!(iss >> e.id >> e.type >> numTags))
                continue;
            for (int i = 0; i < numTags; ++i) {
                int tmp; iss >> tmp;
            }
            int nodeId;
            while (iss >> nodeId)
                e.node_ids.push_back(nodeId);
            elements.push_back(e);
        }
    }
}

//Public API: Parse mesh file 
bool parseMeshFile(const std::string& filename, 
                   std::vector<Node>& nodes, 
                   std::vector<Element>& elements) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }

    // detect version
    std::string line;
    std::getline(file, line);
    if (line != "$MeshFormat") {
        std::cerr << "Not a valid Gmsh mesh file\n";
        file.close();
        return false;
    }
    std::getline(file, line);
    bool isV4 = line.find("4.") != std::string::npos;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // skip $EndMeshFormat

    nodes.clear();
    elements.clear();

    if (isV4) {
        parseMSHv4(file, nodes, elements);
    } else {
        parseMSHv2(file, nodes, elements);
    }
    file.close();
    return true;
}

// Gmsh v4 Parser 
void parseMSHv4(std::ifstream &file, std::vector<Node> &nodes, std::vector<Element> &elements) {
    std::string line;
    while (std::getline(file, line)) {
        if (line == "$Nodes") {
            size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
            file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            nodes.reserve(numNodes);
            for (size_t b = 0; b < numEntityBlocks; ++b) {
                int entityDim, entityTag;
                size_t parametric, numNodesInBlock;
                file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

                std::vector<int> ids(numNodesInBlock);
                for (size_t i = 0; i < numNodesInBlock; ++i)
                    file >> ids[i];

                for (size_t i = 0; i < numNodesInBlock; ++i) {
                    double x, y, z;
                    file >> x >> y >> z;
                    nodes.push_back({ids[i], x, y, z});
                }
            }
            std::getline(file, line); // consume $EndNodes
        }

        if (line == "$Elements") {
            size_t numEntityBlocks, numElements, minElemTag, maxElemTag;
            file >> numEntityBlocks >> numElements >> minElemTag >> maxElemTag;
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            elements.reserve(numElements);
            int elemCounter = 1;
            for (size_t b = 0; b < numEntityBlocks; ++b) {
                int entityDim, entityTag, elemType;
                size_t numElemsInBlock;
                file >> entityDim >> entityTag >> elemType >> numElemsInBlock;

                // Determine number of nodes per element based on element type
                int nodesPerElem = 0;
                if (elemType == 1) nodesPerElem = 2;      // Line
                else if (elemType == 2) nodesPerElem = 3; // Triangle
                else if (elemType == 4) nodesPerElem = 4; // Tet4
                else if (elemType == 5) nodesPerElem = 8; // Hexahedron
                // Add more types as needed
                
                if (nodesPerElem == 0) {
                    std::cerr << "Warning: Unsupported element type " << elemType 
                              << " in block " << b << ". Skipping " << numElemsInBlock 
                              << " elements." << std::endl;
                    // Skip this block by reading and discarding all element lines
                    for (size_t i = 0; i < numElemsInBlock; ++i) {
                        std::string elemLine;
                        std::getline(file, elemLine);
                    }
                    continue;
                }
                
                for (size_t i = 0; i < numElemsInBlock; ++i) {
                    Element e;
                    e.id = elemCounter++;
                    e.type = elemType;

                    // In Gmsh v4, each element line starts with elementTag, then node IDs
                    int elementTag;  // Read and discard the element tag
                    file >> elementTag;
                    
                    // Read node connectivity for this element
                    int nodeId;
                    for (int n = 0; n < nodesPerElem; ++n) {
                        file >> nodeId;
                        e.node_ids.push_back(nodeId);
                    }
                    elements.push_back(e);
                }
            }
            std::getline(file, line); // consume $EndElements
        }
    }
}

// Save to CSV 
void saveCSV(const std::vector<Node> &nodes,
             const std::vector<Element> &elements) {
    std::ofstream nodeCSV("nodes.csv");
    nodeCSV << "id,x,y,z\n";
    for (const auto &n : nodes)
        nodeCSV << n.id << "," << n.x << "," << n.y << "," << n.z << "\n";
    nodeCSV.close();

    std::ofstream elemCSV("elements.csv");
    elemCSV << "id,type,connectivity\n";
    for (const auto &e : elements) {
        elemCSV << e.id << "," << e.type << ",";
        for (size_t i = 0; i < e.node_ids.size(); ++i) {
            elemCSV << e.node_ids[i];
            if (i + 1 < e.node_ids.size()) elemCSV << " ";
        }
        elemCSV << "\n";
    }
    elemCSV.close();
}

// Save to Binary
void saveBinary(const std::vector<Node> &nodes,
                const std::vector<Element> &elements) {
    std::ofstream nodeBin("nodes.bin", std::ios::binary);
    size_t nCount = nodes.size();
    nodeBin.write(reinterpret_cast<const char*>(&nCount), sizeof(size_t));
    for (const auto &n : nodes)
        nodeBin.write(reinterpret_cast<const char*>(&n), sizeof(Node));
    nodeBin.close();

    std::ofstream elemBin("elements.bin", std::ios::binary);
    size_t eCount = elements.size();
    elemBin.write(reinterpret_cast<const char*>(&eCount), sizeof(size_t));
    for (const auto &e : elements) {
        elemBin.write(reinterpret_cast<const char*>(&e.id), sizeof(int));
        elemBin.write(reinterpret_cast<const char*>(&e.type), sizeof(int));
        size_t connSize = e.node_ids.size();
        elemBin.write(reinterpret_cast<const char*>(&connSize), sizeof(size_t));
        elemBin.write(reinterpret_cast<const char*>(e.node_ids.data()), connSize * sizeof(int));
    }
    elemBin.close();
}

