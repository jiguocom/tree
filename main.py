from common.DecisionTree import DecisionTree
import copy

insert_log = "开始匹配数据"
dispose_state = 2

def print_exit(val):
    print(val)
    exit(0)

def clearStr(strs):
    return "".join(strs.split())

# 获取产品决策模型
def getProductTree():

    product_train_data = [
        ["【7天修复24小时水润】法国维德勒修护护手膏温和植萃清香补水滋润不油腻","1447"],
        ["【柔软舒适】年纪法莱绒乐肤棉四件套","1539"],
        ["【双重保暖】HEATWING循环热阻科技保暖内衣裤","1545"],
        ["德国Sicherheit马桶垫","279"],
        ["【一搓即香爆香黑科技】PWU双色衣物留香珠双色留香柔顺护衣","1453"],
        ["爆仓72小时发货【专利设计暖而不燥】Kehealk2科西取暖器远程遥控暖随心意“优选”","1502"],
    ]

    # 获取决策树
    labels = ["产品名"]
    tree = DecisionTree()
    myTree = tree.createTree(product_train_data, copy.copy(labels))
    return myTree

# 测试数据
def productProcess(prouctTree, testProductName):

    # 格式化测试数据
    testProductName = clearStr(testProductName)
    tree = DecisionTree()
    labels = ["产品名"]

    return tree.classify(prouctTree, labels, [testProductName])

if __name__=='__main__':
    tree = getProductTree()
    print_exit(productProcess(tree, "德国Sicherheit马桶垫"))