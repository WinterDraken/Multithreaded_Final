#ifndef MESH_PARSER_H
#define MESH_PARSER_H

#include <vector>
#include <string>
#include <fstream>

struct Node {
    int id;
    double x, y, z;
};

struct Element {
    int id;
    int type;
    std::vector<int> node_ids;
};

// Parse Gmsh mesh file (v2 or v4 format)
// Returns true on success, false on failure
bool parseMeshFile(const std::string& filename, 
                   std::vector<Node>& nodes, 
                   std::vector<Element>& elements);

// Internal parsers (called by parseMeshFile)
void parseMSHv2(std::ifstream &file, std::vector<Node> &nodes, std::vector<Element> &elements);
void parseMSHv4(std::ifstream &file, std::vector<Node> &nodes, std::vector<Element> &elements);

#endif // MESH_PARSER_H

