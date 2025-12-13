<template>
  <div class="lora-manager">
    <div class="page-header">
      <h1><el-icon><Files /></el-icon> LoRA 模型管理</h1>
      <el-button @click="fetchLoras" :loading="loading">
        <el-icon><Refresh /></el-icon> 刷新
      </el-button>
    </div>

    <el-card class="lora-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <span>训练产出 ({{ loraList.length }} 个模型)</span>
            <!-- 批量操作按钮 -->
            <div class="batch-actions" v-if="selectedLoras.length > 0">
              <el-button type="primary" size="small" @click="batchDownload">
                <el-icon><Download /></el-icon> 下载选中 ({{ selectedLoras.length }})
              </el-button>
              <el-button type="danger" size="small" @click="batchDelete">
                <el-icon><Delete /></el-icon> 删除选中 ({{ selectedLoras.length }})
              </el-button>
            </div>
          </div>
          <div class="path-hint">
            <span>路径: {{ loraPath }}</span>
            <el-button 
              type="primary" 
              link 
              size="small" 
              @click="copyPath"
              class="copy-btn"
            >
              <el-icon><CopyDocument /></el-icon>
            </el-button>
          </div>
        </div>
      </template>

      <div v-loading="loading" class="lora-content">
        <el-empty v-if="loraList.length === 0 && !loading" description="暂无 LoRA 模型">
          <template #image>
            <el-icon style="font-size: 64px; color: var(--el-text-color-secondary)"><FolderOpened /></el-icon>
          </template>
        </el-empty>

        <el-table 
          v-else 
          :data="loraList" 
          style="width: 100%" 
          stripe
          @selection-change="handleSelectionChange"
          ref="tableRef"
        >
          <el-table-column type="selection" width="50" />
          
          <el-table-column prop="name" label="文件名" min-width="300">
            <template #default="{ row }">
              <div class="file-name">
                <el-icon class="file-icon"><Document /></el-icon>
                <span>{{ row.name }}</span>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column prop="size" label="大小" width="120" align="right">
            <template #default="{ row }">
              {{ formatSize(row.size) }}
            </template>
          </el-table-column>

          <el-table-column label="操作" width="200" align="center">
            <template #default="{ row }">
              <el-button-group>
                <el-button type="primary" size="small" @click="downloadLora(row)">
                  <el-icon><Download /></el-icon> 下载
                </el-button>
                <el-button type="danger" size="small" @click="deleteLora(row)">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </el-button-group>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <!-- 删除确认对话框（单个） -->
    <el-dialog v-model="deleteDialogVisible" title="确认删除" width="400px">
      <p>确定要删除 LoRA 模型吗？</p>
      <p class="delete-filename">{{ selectedLora?.name }}</p>
      <p class="warning-text">此操作不可恢复！</p>
      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmDelete" :loading="deleting">删除</el-button>
      </template>
    </el-dialog>

    <!-- 批量删除确认对话框 -->
    <el-dialog v-model="batchDeleteDialogVisible" title="批量删除确认" width="500px">
      <p>确定要删除以下 {{ selectedLoras.length }} 个 LoRA 模型吗？</p>
      <div class="batch-delete-list">
        <div v-for="lora in selectedLoras" :key="lora.path" class="delete-item">
          <el-icon><Document /></el-icon>
          <span>{{ lora.name }}</span>
          <span class="delete-item-size">{{ formatSize(lora.size) }}</span>
        </div>
      </div>
      <p class="warning-text">⚠️ 此操作不可恢复！</p>
      <template #footer>
        <el-button @click="batchDeleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmBatchDelete" :loading="deleting">
          删除全部 ({{ selectedLoras.length }})
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Files, Refresh, Document, Download, Delete, FolderOpened, CopyDocument } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

interface LoraItem {
  name: string
  path: string
  size: number
}

const loading = ref(false)
const loraList = ref<LoraItem[]>([])
const loraPath = ref('')
const deleteDialogVisible = ref(false)
const batchDeleteDialogVisible = ref(false)
const selectedLora = ref<LoraItem | null>(null)
const selectedLoras = ref<LoraItem[]>([])
const deleting = ref(false)
const tableRef = ref()

const formatSize = (bytes: number) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}

const copyPath = async () => {
  try {
    await navigator.clipboard.writeText(loraPath.value)
    ElMessage.success('路径已复制到剪贴板')
  } catch (e) {
    // 降级方案
    const textarea = document.createElement('textarea')
    textarea.value = loraPath.value
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
    ElMessage.success('路径已复制到剪贴板')
  }
}

const fetchLoras = async () => {
  loading.value = true
  selectedLoras.value = []
  try {
    const res = await axios.get('/api/loras')
    // 新的返回格式: { loras, loraPath, loraPathExists }
    loraList.value = res.data.loras || res.data || []
    loraPath.value = res.data.loraPath || './output'
    
    // 调试日志
    console.log('[LoRA] loraPath:', res.data.loraPath )
    console.log('[LoRA] loraPathExists:', res.data.loraPathExists)
    console.log('[LoRA] loras count:', loraList.value.length)
  } catch (e) {
    console.error('Failed to fetch LoRAs:', e)
    ElMessage.error('获取 LoRA 列表失败')
  } finally {
    loading.value = false
  }
}

// 多选变化处理
const handleSelectionChange = (selection: LoraItem[]) => {
  selectedLoras.value = selection
}

// 单个下载
const downloadLora = (lora: LoraItem) => {
  const link = document.createElement('a')
  link.href = `/api/loras/download?path=${encodeURIComponent(lora.path)}`
  link.setAttribute('download', lora.name.split('/').pop() || 'lora.safetensors')
  document.body.appendChild(link)
  link.click()
  link.remove()
  ElMessage.info('已开始下载')
}

// 批量下载
const batchDownload = () => {
  if (selectedLoras.value.length === 0) return
  
  ElMessage.info(`开始下载 ${selectedLoras.value.length} 个文件...`)
  
  // 逐个触发下载（浏览器会自动处理多个下载）
  selectedLoras.value.forEach((lora, index) => {
    setTimeout(() => {
      downloadLora(lora)
    }, index * 500) // 间隔 500ms 避免浏览器阻止
  })
}

// 单个删除
const deleteLora = (lora: LoraItem) => {
  selectedLora.value = lora
  deleteDialogVisible.value = true
}

// 批量删除确认
const batchDelete = () => {
  if (selectedLoras.value.length === 0) return
  batchDeleteDialogVisible.value = true
}

// 确认单个删除
const confirmDelete = async () => {
  if (!selectedLora.value) return
  
  deleting.value = true
  try {
    await axios.delete(`/api/loras/delete?path=${encodeURIComponent(selectedLora.value.path)}`)
    ElMessage.success('删除成功')
    deleteDialogVisible.value = false
    fetchLoras()
  } catch (e) {
    console.error('Delete failed:', e)
    ElMessage.error('删除失败')
  } finally {
    deleting.value = false
  }
}

// 确认批量删除
const confirmBatchDelete = async () => {
  if (selectedLoras.value.length === 0) return
  
  deleting.value = true
  let successCount = 0
  let failCount = 0
  
  try {
    for (const lora of selectedLoras.value) {
      try {
        await axios.delete(`/api/loras/delete?path=${encodeURIComponent(lora.path)}`)
        successCount++
      } catch (e) {
        console.error('Delete failed:', lora.name, e)
        failCount++
      }
    }
    
    if (failCount === 0) {
      ElMessage.success(`成功删除 ${successCount} 个文件`)
    } else {
      ElMessage.warning(`删除完成: ${successCount} 成功, ${failCount} 失败`)
    }
    
    batchDeleteDialogVisible.value = false
    fetchLoras()
  } catch (e) {
    console.error('Batch delete error:', e)
    ElMessage.error('批量删除出错')
  } finally {
    deleting.value = false
  }
}

onMounted(() => {
  fetchLoras()
})
</script>

<style scoped>
.lora-manager {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h1 {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 24px;
  margin: 0;
}

.lora-card {
  background: var(--el-bg-color);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.batch-actions {
  display: flex;
  gap: 8px;
}

.path-hint {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
}

.copy-btn {
  padding: 2px 4px;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.copy-btn:hover {
  opacity: 1;
}

.lora-content {
  min-height: 300px;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-icon {
  color: var(--el-color-primary);
}

.delete-filename {
  font-family: monospace;
  background: var(--el-fill-color-light);
  padding: 8px 12px;
  border-radius: 4px;
  word-break: break-all;
}

.warning-text {
  color: var(--el-color-danger);
  font-size: 12px;
  margin-top: 12px;
}

.batch-delete-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-light);
  border-radius: 4px;
  margin: 12px 0;
}

.delete-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  font-size: 13px;
}

.delete-item:last-child {
  border-bottom: none;
}

.delete-item .el-icon {
  color: var(--el-color-primary);
  flex-shrink: 0;
}

.delete-item span {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delete-item-size {
  flex: none !important;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

:deep(.el-table) {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
}

:deep(.el-table .el-table__header-wrapper th) {
  background: var(--el-fill-color-light);
}
</style>
