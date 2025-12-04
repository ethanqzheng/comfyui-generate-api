"""
智能工作流分析器 - 基于ComfyUI图执行原理的深度解析
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """节点角色枚举"""
    MODEL_LOADER = "model_loader"      # 模型加载器
    POSITIVE_PROMPT = "positive_prompt"  # 正面提示词
    NEGATIVE_PROMPT = "negative_prompt"  # 负面提示词
    LATENT_SOURCE = "latent_source"    # 潜在空间源
    IMAGE_INPUT = "image_input"        # 图像输入（图生图）
    SAMPLER = "sampler"                # 采样器
    DECODER = "decoder"                # 解码器
    OUTPUT = "output"                  # 输出节点
    PREPROCESSOR = "preprocessor"      # 预处理器
    POSTPROCESSOR = "postprocessor"    # 后处理器
    CONTROL = "control"                # 控制节点
    UNKNOWN = "unknown"                # 未知角色


@dataclass
class NodeConnection:
    """节点连接信息"""
    from_node: str
    from_output: int
    to_node: str
    to_input: str
    data_type: Optional[str] = None


@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str
    class_type: str
    inputs: Dict[str, Any]
    role: NodeRole
    title: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    connections_in: List[NodeConnection] = None
    connections_out: List[NodeConnection] = None
    
    def __post_init__(self):
        if self.connections_in is None:
            self.connections_in = []
        if self.connections_out is None:
            self.connections_out = []


@dataclass
class WorkflowGraph:
    """工作流图结构"""
    nodes: Dict[str, WorkflowNode]
    connections: List[NodeConnection]
    execution_order: List[str]
    input_parameters: Dict[str, Any]
    output_nodes: List[str]
    workflow_type: str
    
    def get_node_by_role(self, role: NodeRole) -> List[WorkflowNode]:
        """根据角色获取节点"""
        return [node for node in self.nodes.values() if node.role == role]
    
    def get_prompt_nodes(self) -> Tuple[Optional[WorkflowNode], Optional[WorkflowNode]]:
        """获取正负面提示词节点"""
        positive = self.get_node_by_role(NodeRole.POSITIVE_PROMPT)
        negative = self.get_node_by_role(NodeRole.NEGATIVE_PROMPT)
        return (positive[0] if positive else None, negative[0] if negative else None)
    
    def get_image_input_node(self) -> Optional[WorkflowNode]:
        """获取图像输入节点（LoadImage）"""
        image_input = self.get_node_by_role(NodeRole.IMAGE_INPUT)
        return image_input[0] if image_input else None


class WorkflowAnalyzer:
    """智能工作流分析器"""
    
    # 节点类型到角色的映射
    NODE_TYPE_ROLES = {
        'CheckpointLoaderSimple': NodeRole.MODEL_LOADER,
        'CheckpointLoader': NodeRole.MODEL_LOADER,
        'LoraLoader': NodeRole.MODEL_LOADER,
        'VAELoader': NodeRole.MODEL_LOADER,
        'CLIPLoader': NodeRole.MODEL_LOADER,
        
        'CLIPTextEncode': NodeRole.UNKNOWN,  # 需要进一步分析
        'CLIPTextEncodeSDXL': NodeRole.UNKNOWN,
        
        'EmptyLatentImage': NodeRole.LATENT_SOURCE,
        'LatentUpscale': NodeRole.LATENT_SOURCE,
        'LoadImage': NodeRole.IMAGE_INPUT,
        'VAEEncode': NodeRole.LATENT_SOURCE,
        
        'KSampler': NodeRole.SAMPLER,
        'KSamplerAdvanced': NodeRole.SAMPLER,
        'SamplerCustom': NodeRole.SAMPLER,
        
        'VAEDecode': NodeRole.DECODER,
        'VAEDecodeTiled': NodeRole.DECODER,
        
        'SaveImage': NodeRole.OUTPUT,
        'PreviewImage': NodeRole.OUTPUT,
        'ImageSave': NodeRole.OUTPUT,
        
        'ControlNetApply': NodeRole.CONTROL,
        'ControlNetLoader': NodeRole.CONTROL,
        'IPAdapterApply': NodeRole.CONTROL,
        
        'ImageScale': NodeRole.POSTPROCESSOR,
        'ImageUpscaleWithModel': NodeRole.POSTPROCESSOR,
        'ImageCrop': NodeRole.POSTPROCESSOR,
    }
    
    # 数据类型映射
    DATA_TYPES = {
        'MODEL': 'model',
        'CLIP': 'clip', 
        'VAE': 'vae',
        'CONDITIONING': 'conditioning',
        'LATENT': 'latent',
        'IMAGE': 'image',
        'MASK': 'mask',
        'STRING': 'string',
        'INT': 'int',
        'FLOAT': 'float',
        'BOOLEAN': 'boolean',
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowGraph:
        """分析工作流，返回结构化的图表示"""
        
        # 1. 解析节点
        nodes = self._parse_nodes(workflow_data)
        
        # 2. 分析连接关系
        connections = self._analyze_connections(workflow_data, nodes)
        
        # 3. 确定节点角色
        self._determine_node_roles(nodes, connections)
        
        # 4. 计算执行顺序
        execution_order = self._calculate_execution_order(nodes, connections)
        
        # 5. 提取输入参数
        input_parameters = self._extract_input_parameters(nodes)
        
        # 6. 识别输出节点
        output_nodes = self._identify_output_nodes(nodes)
        
        # 7. 确定工作流类型
        workflow_type = self._determine_workflow_type(nodes)
        
        return WorkflowGraph(
            nodes=nodes,
            connections=connections,
            execution_order=execution_order,
            input_parameters=input_parameters,
            output_nodes=output_nodes,
            workflow_type=workflow_type
        )
    
    def _parse_nodes(self, workflow_data: Dict[str, Any]) -> Dict[str, WorkflowNode]:
        """解析节点"""
        nodes = {}
        
        for node_id, node_data in workflow_data.items():
            if not isinstance(node_data, dict) or 'class_type' not in node_data:
                continue
            
            class_type = node_data['class_type']
            inputs = node_data.get('inputs', {})
            meta = node_data.get('_meta', {})
            title = meta.get('title', class_type)
            
            # 初始角色分配
            role = self.NODE_TYPE_ROLES.get(class_type, NodeRole.UNKNOWN)
            
            node = WorkflowNode(
                id=node_id,
                class_type=class_type,
                inputs=inputs,
                role=role,
                title=title,
                meta=meta
            )
            
            nodes[node_id] = node
        
        return nodes
    
    def _analyze_connections(self, workflow_data: Dict[str, Any], nodes: Dict[str, WorkflowNode]) -> List[NodeConnection]:
        """分析节点连接关系"""
        connections = []
        
        for node_id, node_data in workflow_data.items():
            if node_id not in nodes:
                continue
            
            inputs = node_data.get('inputs', {})
            
            for input_name, input_value in inputs.items():
                if self._is_link(input_value):
                    from_node, from_output = input_value[0], input_value[1]
                    
                    connection = NodeConnection(
                        from_node=from_node,
                        from_output=from_output,
                        to_node=node_id,
                        to_input=input_name
                    )
                    
                    connections.append(connection)
                    
                    # 添加到节点的连接列表
                    if from_node in nodes:
                        nodes[from_node].connections_out.append(connection)
                    nodes[node_id].connections_in.append(connection)
        
        return connections
    
    def _is_link(self, value) -> bool:
        """判断是否为节点连接"""
        return (isinstance(value, list) and 
                len(value) == 2 and 
                isinstance(value[0], str) and 
                isinstance(value[1], (int, float)))
    
    def _determine_node_roles(self, nodes: Dict[str, WorkflowNode], connections: List[NodeConnection]):
        """确定节点角色，特别是CLIPTextEncode节点"""
        
        # 找到采样器节点
        sampler_nodes = [node for node in nodes.values() if node.role == NodeRole.SAMPLER]
        
        for sampler in sampler_nodes:
            # 分析采样器的输入连接
            for conn in sampler.connections_in:
                if conn.to_input == 'positive':
                    # 正面提示词
                    if conn.from_node in nodes:
                        nodes[conn.from_node].role = NodeRole.POSITIVE_PROMPT
                elif conn.to_input == 'negative':
                    # 负面提示词
                    if conn.from_node in nodes:
                        nodes[conn.from_node].role = NodeRole.NEGATIVE_PROMPT
        
        # 对于没有连接到采样器的CLIPTextEncode节点，通过文本内容判断
        for node in nodes.values():
            if node.class_type == 'CLIPTextEncode' and node.role == NodeRole.UNKNOWN:
                text_content = str(node.inputs.get('text', '')).lower()
                if self._is_negative_text(text_content):
                    node.role = NodeRole.NEGATIVE_PROMPT
                else:
                    node.role = NodeRole.POSITIVE_PROMPT
    
    def _is_negative_text(self, text: str) -> bool:
        """判断文本是否为负面提示词"""
        negative_keywords = [
            'low quality', 'blurry', 'nsfw', 'bad', 'ugly', 'worst quality',
            'low res', 'error', 'cropped', 'jpeg artifacts', 'signature',
            'watermark', 'username', 'artist name', 'trademark', 'title',
            'multiple view', 'reference sheet', 'long body', 'malformed',
            'poorly drawn', 'bad anatomy', 'disfigured', 'mutated', 'extra',
            'missing', 'floating', 'disconnected', 'distorted', 'duplicate',
            'mutation', 'deformed', 'gross proportions', 'missing arms',
            'missing legs', 'extra arms', 'extra legs', 'fused fingers',
            'too many fingers', 'long neck', 'cross-eyed'
        ]
        
        return any(keyword in text for keyword in negative_keywords)
    
    def _calculate_execution_order(self, nodes: Dict[str, WorkflowNode], connections: List[NodeConnection]) -> List[str]:
        """计算执行顺序（拓扑排序）"""
        # 构建依赖图
        dependencies = {node_id: set() for node_id in nodes.keys()}
        
        for conn in connections:
            dependencies[conn.to_node].add(conn.from_node)
        
        # 拓扑排序
        execution_order = []
        remaining = set(nodes.keys())
        
        while remaining:
            # 找到没有依赖的节点
            ready = [node_id for node_id in remaining 
                    if not dependencies[node_id] or 
                    dependencies[node_id].issubset(set(execution_order))]
            
            if not ready:
                # 循环依赖，选择第一个剩余节点
                ready = [next(iter(remaining))]
            
            # 优先选择输出节点
            output_ready = [node_id for node_id in ready if nodes[node_id].role == NodeRole.OUTPUT]
            if output_ready:
                next_node = output_ready[0]
            else:
                next_node = ready[0]
            
            execution_order.append(next_node)
            remaining.remove(next_node)
        
        return execution_order
    
    def _extract_input_parameters(self, nodes: Dict[str, WorkflowNode]) -> Dict[str, Any]:
        """提取可配置的输入参数"""
        parameters = {}
        
        # 提示词参数
        positive_nodes = [n for n in nodes.values() if n.role == NodeRole.POSITIVE_PROMPT]
        negative_nodes = [n for n in nodes.values() if n.role == NodeRole.NEGATIVE_PROMPT]
        
        if positive_nodes:
            parameters['prompt'] = {
                'type': 'string',
                'description': '正面提示词',
                'default': positive_nodes[0].inputs.get('text', ''),
                'node_id': positive_nodes[0].id,
                'input_name': 'text'
            }
        
        if negative_nodes:
            parameters['negative_prompt'] = {
                'type': 'string', 
                'description': '负面提示词',
                'default': negative_nodes[0].inputs.get('text', ''),
                'node_id': negative_nodes[0].id,
                'input_name': 'text'
            }
        
        # 采样器参数
        sampler_nodes = [n for n in nodes.values() if n.role == NodeRole.SAMPLER]
        if sampler_nodes:
            sampler = sampler_nodes[0]
            sampler_inputs = sampler.inputs
            
            if 'steps' in sampler_inputs:
                parameters['steps'] = {
                    'type': 'integer',
                    'description': '采样步数',
                    'default': sampler_inputs['steps'],
                    'min': 1,
                    'max': 100,
                    'node_id': sampler.id,
                    'input_name': 'steps'
                }
            
            if 'cfg' in sampler_inputs:
                parameters['cfg_scale'] = {
                    'type': 'float',
                    'description': 'CFG Scale',
                    'default': sampler_inputs['cfg'],
                    'min': 1.0,
                    'max': 20.0,
                    'node_id': sampler.id,
                    'input_name': 'cfg'
                }
            
            if 'seed' in sampler_inputs:
                parameters['seed'] = {
                    'type': 'integer',
                    'description': '随机种子',
                    'default': sampler_inputs['seed'],
                    'node_id': sampler.id,
                    'input_name': 'seed'
                }
            
            if 'sampler_name' in sampler_inputs:
                parameters['sampler_name'] = {
                    'type': 'string',
                    'description': '采样器类型',
                    'default': sampler_inputs['sampler_name'],
                    'node_id': sampler.id,
                    'input_name': 'sampler_name'
                }
        
        # 图像尺寸参数
        latent_nodes = [n for n in nodes.values() if n.role == NodeRole.LATENT_SOURCE and n.class_type == 'EmptyLatentImage']
        if latent_nodes:
            latent = latent_nodes[0]
            latent_inputs = latent.inputs
            
            if 'width' in latent_inputs:
                parameters['width'] = {
                    'type': 'integer',
                    'description': '图像宽度',
                    'default': latent_inputs['width'],
                    'min': 64,
                    'max': 2048,
                    'node_id': latent.id,
                    'input_name': 'width'
                }
            
            if 'height' in latent_inputs:
                parameters['height'] = {
                    'type': 'integer',
                    'description': '图像高度',
                    'default': latent_inputs['height'],
                    'min': 64,
                    'max': 2048,
                    'node_id': latent.id,
                    'input_name': 'height'
                }
            
            if 'batch_size' in latent_inputs:
                parameters['batch_size'] = {
                    'type': 'integer',
                    'description': '批次大小',
                    'default': latent_inputs['batch_size'],
                    'min': 1,
                    'max': 8,
                    'node_id': latent.id,
                    'input_name': 'batch_size'
                }
        
        # 图生图参数
        image_input_nodes = [n for n in nodes.values() if n.role == NodeRole.IMAGE_INPUT]
        if image_input_nodes:
            image_input = image_input_nodes[0]
            image_inputs = image_input.inputs
            
            if 'image' in image_inputs:
                parameters['input_image'] = {
                    'type': 'string',
                    'description': '输入图像文件名',
                    'default': image_inputs['image'],
                    'node_id': image_input.id,
                    'input_name': 'image'
                }
        
        # denoise参数（通常在KSampler节点中）
        if sampler_nodes:
            sampler = sampler_nodes[0]
            if 'denoise' in sampler.inputs:
                parameters['denoise'] = {
                    'type': 'float',
                    'description': '降噪强度（图生图）',
                    'default': sampler.inputs['denoise'],
                    'min': 0.0,
                    'max': 1.0,
                    'node_id': sampler.id,
                    'input_name': 'denoise'
                }
        
        return parameters
    
    def _identify_output_nodes(self, nodes: Dict[str, WorkflowNode]) -> List[str]:
        """识别输出节点"""
        return [node.id for node in nodes.values() if node.role == NodeRole.OUTPUT]
    
    def _determine_workflow_type(self, nodes: Dict[str, WorkflowNode]) -> str:
        """确定工作流类型"""
        node_types = [node.class_type for node in nodes.values()]
        
        if 'LoadImage' in node_types and 'KSampler' in node_types:
            if 'ControlNetApply' in node_types:
                return 'controlnet'
            else:
                return 'img2img'
        elif 'EmptyLatentImage' in node_types and 'KSampler' in node_types:
            return 'text2img'
        elif any('Upscale' in t for t in node_types):
            return 'upscale'
        else:
            return 'custom'
    
    def get_parameter_update_mapping(self, graph: WorkflowGraph) -> Dict[str, Dict[str, str]]:
        """获取参数更新映射"""
        mapping = {}
        
        for param_name, param_info in graph.input_parameters.items():
            if 'node_id' in param_info and 'input_name' in param_info:
                node_id = param_info['node_id']
                input_name = param_info['input_name']
                
                if node_id not in mapping:
                    mapping[node_id] = {}
                mapping[node_id][input_name] = param_name
        
        return mapping
    
    def print_analysis(self, graph: WorkflowGraph):
        """打印分析结果"""
        print(f"\n=== 工作流分析结果 ===")
        print(f"工作流类型: {graph.workflow_type}")
        print(f"节点数量: {len(graph.nodes)}")
        print(f"连接数量: {len(graph.connections)}")
        
        print(f"\n=== 节点角色分布 ===")
        role_counts = {}
        for node in graph.nodes.values():
            role_counts[node.role] = role_counts.get(node.role, 0) + 1
        
        for role, count in role_counts.items():
            print(f"{role.value}: {count}")
        
        print(f"\n=== 输入参数 ===")
        for param_name, param_info in graph.input_parameters.items():
            print(f"{param_name}: {param_info['description']} (默认: {param_info['default']})")
        
        print(f"\n=== 执行顺序 ===")
        for i, node_id in enumerate(graph.execution_order):
            node = graph.nodes[node_id]
            print(f"{i+1}. {node_id}: {node.class_type} ({node.role.value})")
